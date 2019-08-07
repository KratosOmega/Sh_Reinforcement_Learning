from logging import getLogger
logger = getLogger(__name__)

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import get_variables
import gc
import datetime

from .utils import get_timestamp

class NAF(object):
  def __init__(self, sess,
               env, strategy, pred_network, target_network, stat,
               discount, batch_size, learning_rate,
               max_steps, update_repeat, max_episodes):
    self.sess = sess
    self.env = env
    self.strategy = strategy
    self.pred_network = pred_network
    self.target_network = target_network
    self.stat = stat

    self.discount = discount
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.action_size = env.action_space.shape[0]

    self.max_steps = max_steps
    self.update_repeat = update_repeat
    self.max_episodes = max_episodes

    self.prestates = []
    self.actions = []
    self.rewards = []
    self.poststates = []
    self.terminals = []
    # the initial speed give to the beginning of the traffic
    self.speed_input = [75, 75, 75]
    #self.termination_threshold = 0.95

    with tf.name_scope('optimizer'):
      self.target_y = tf.compat.v1.placeholder(tf.float32, [None], name='target_y')
      self.loss = tf.reduce_mean(tf.math.squared_difference(self.target_y, tf.squeeze(self.pred_network.Q)), name='loss')

      self.optim = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

  def run(self, is_train=True):
    print("------------------------------------ 1")

    self.stat.load_model()
    self.target_network.hard_copy_from(self.pred_network)

    for self.idx_episode in range(self.max_episodes):
      state = self.env.reset(self.speed_input)
      cumulative_r = 0

      for t in range(0, self.max_steps):
        # 1. predict
        action = self.predict(state)

        # 2. step
        self.prestates.append(state)

        # transform actions into vissim version
        action = np.clip(action, -1, 1)
        transformed_action = self.convert_actions(action)

        state, reward, terminal = self.env.step(transformed_action)

        cumulative_r += reward

        print("---------------------------" + str(t))
        print(state)
        print(transformed_action)
        print(reward)
        print(cumulative_r / (t + 1))
        print("")

        self.poststates.append(state)

        # ----------------------------------------------------------------------- termination logic block
        """
        # using only one desired reward to terminate
        terminal = True if t == self.max_steps - 1 else terminal
        """

        # using only average desired reward to terminate
        #if t == self.max_steps - 1 or (cumulative_r / (t + 1) > self.termination_threshold and t > 10):
        if t == self.max_steps - 1:
          terminal = True
        else:
          terminal = False
        # -----------------------------------------------------------------------

        # 3. perceive
        if is_train:
          q, v, a, l = self.perceive(state, reward, action, terminal)

          if self.stat:
            self.stat.on_step(action, reward, terminal, q, v, a, l)

        if terminal:
          self.strategy.reset()
          break

        # garbage recycling
        gc.collect()

  def run2(self, monitor=False, display=False, is_train=True):
    print("------------------------------------ 2")

    target_y = tf.placeholder(tf.float32, [None], name='target_y')
    loss = tf.reduce_mean(tf.squared_difference(target_y, tf.squeeze(self.pred_network.Q)), name='loss')

    optim = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    self.stat.load_model()
    self.target_network.hard_copy_from(self.pred_network)

    # replay memory
    prestates = []
    actions = []
    rewards = []
    poststates = []
    terminals = []

    # the main learning loop
    total_reward = 0
    for i_episode in range(self.max_episodes):
      state = self.env.reset(self.speed_input)
      episode_reward = 0
      loss_ = 0

      for t in range(self.max_steps):
        # predict the mean action from current state
        x_ = np.array([state])
        u_ = self.pred_network.mu.eval({self.pred_network.x: x_})[0]

        action = u_ + np.random.randn(1) / (i_episode + 1)

        prestates.append(state)
        actions.append(action)

        # transform actions into vissim version
        action = np.clip(action, -1, 1)
        transformed_action = self.convert_actions(action)
        state, reward, terminal = self.env.step(transformed_action)
        episode_reward += reward

        print("---------------------------" + str(t))
        print(state)
        print(transformed_action)
        print(reward)
        print(episode_reward / (t + 1))
        print("")

        rewards.append(reward); poststates.append(state); terminals.append(terminal)

        if len(prestates) > 10:
          loss_ = 0
          for k in range(self.update_repeat):
            if len(prestates) > self.batch_size:
              indexes = np.random.choice(len(prestates), size=self.batch_size)
            else:
              indexes = range(len(prestates))

            # Q-update
            v_ = self.target_network.V.eval({self.target_network.x: np.array(poststates)[indexes]})
            y_ = np.array(rewards)[indexes] + self.discount * np.squeeze(v_)

            tmp1, tmp2 = np.array(prestates)[indexes], np.array(actions)[indexes]
            loss_ += l_

            self.target_network.soft_update_from(self.pred_network)

            print("average loss:", loss_/k)
            print("Episode {} finished after {} timesteps, reward {}".format(i_episode + 1, t + 1, episode_reward))

        if terminal:
          break

        # garbage recycling
        gc.collect()

      total_reward += episode_reward

    print("Average reward per episode {}".format(total_reward / self.episodes))

  def predict(self, state):
    u = self.pred_network.predict([state])[0]

    return self.strategy.add_noise(u, {'idx_episode': self.idx_episode})

  def perceive(self, state, reward, action, terminal):
    self.rewards.append(reward)
    self.actions.append(action)

    return self.q_learning_minibatch()

  def q_learning_minibatch(self):
    q_list = []
    v_list = []
    a_list = []
    l_list = []

    for iteration in range(self.update_repeat):
      if len(self.rewards) >= self.batch_size:
        indexes = np.random.choice(len(self.rewards), size=self.batch_size)
      else:
        indexes = np.arange(len(self.rewards))

      x_t = np.array(self.prestates)[indexes]
      x_t_plus_1 = np.array(self.poststates)[indexes]
      r_t = np.array(self.rewards)[indexes]
      u_t = np.array(self.actions)[indexes]

      v = self.target_network.predict_v(x_t_plus_1, u_t)
      target_y = self.discount * np.squeeze(v) + r_t

      _, l, q, v, a = self.sess.run([
        self.optim, self.loss,
        self.pred_network.Q, self.pred_network.V, self.pred_network.A,
      ], {
        self.target_y: target_y,
        self.pred_network.x: x_t,
        self.pred_network.u: u_t,
        self.pred_network.is_train: True,
      })

      q_list.extend(q)
      v_list.extend(v)
      a_list.extend(a)
      l_list.append(l)

      self.target_network.soft_update_from(self.pred_network)

      logger.debug("q: %s, v: %s, a: %s, l: %s" \
        % (np.mean(q), np.mean(v), np.mean(a), np.mean(l)))

    return np.sum(q_list), np.sum(v_list), np.sum(a_list), np.sum(l_list)

  def convert_actions(self, actions):
        action_space = 19 # [30, 120]
        min_speed = 30
        max_speed = 120
        min_output = -1 # tanh
        #min_output = 0 # sigmoid
        max_output = 1
        output_range = max_output - min_output
        mapping_size = output_range / action_space
        speed_limits = []

        for a in actions:
            action = int((a + 1) / mapping_size) # tanh
            #action = int(a / mapping_size) # sigmoid
            speed = int(30 + action * 5)

            if speed > max_speed:
                speed = max_speed
            if speed < min_speed:
                speed = min_speed
            speed_limits.append(speed)

        return speed_limits
