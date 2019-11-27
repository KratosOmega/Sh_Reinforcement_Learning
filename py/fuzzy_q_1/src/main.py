"""
Author: XIN LI
"""
# ---------------------------------- customized libs
from utils import updateReport, load_data, even_dist, memb_func_eval
# ----------------------------------
# ---------------------------------- public libs
import math
import numpy as np
import random
import gym
from collections import deque
import operator
import matplotlib.pyplot as plt
# ----------------------------------
def presetup_separated():
    input_memb_dim = 50
    output_memb_dim = 9
    action_dim = 3

    ufr_memb_func = even_dist(0, 7000, input_memb_dim)
    dr1_memb_func = even_dist(0, 1.0, input_memb_dim)
    dr2_memb_func = even_dist(0, 1.0, input_memb_dim)
    dr3_memb_func = even_dist(0, 1.0, input_memb_dim)

    qtable = np.zeros((
        input_memb_dim, 
        input_memb_dim, 
        input_memb_dim, 
        input_memb_dim, 
        action_dim, 
        output_memb_dim)
    )

    return ufr_memb_func, dr1_memb_func, dr2_memb_func, dr3_memb_func, qtable

def offline_learning(episode, step, gamma=0.9, alpha=0.5, epsilon=0.1, is_load = False):
    data = load_data("./dataset/data.csv")

    speed_map = [30, 40, 50, 60, 70, 80, 90, 100, 110]
    ufr_memb_func, dr1_memb_func, dr2_memb_func, dr3_memb_func, qtable = presetup_separated()

    if is_load:
        #qtable = np.load(r"\saved\qtable.npy")
        qtable = np.load("./_saved/qtable.npy")
        print("Q-Table is Loaded!")

    for e in range(episode):
        print(" ###########################: ", str(e))
        cum_reward = 0

        steps = len(data)

        for s in range(steps):
            d = data[s]
            state = d[0]
            actions = d[1]
            reward = d[2]
            next_state = d[3]

            print(state, actions, reward, next_state)

            s0 = memb_func_eval(ufr_memb_func, state[0])
            s1 = memb_func_eval(dr1_memb_func, state[1])
            s2 = memb_func_eval(dr2_memb_func, state[2])
            s3 = memb_func_eval(dr3_memb_func, state[3])

            actions_of_state = qtable[s0][s1][s2][s3]

            a0_idx = speed_map.index(actions[0])
            a1_idx = speed_map.index(actions[1])
            a2_idx = speed_map.index(actions[2])

            n_s0 = memb_func_eval(ufr_memb_func, next_state[0])
            n_s1 = memb_func_eval(dr1_memb_func, next_state[1])
            n_s2 = memb_func_eval(dr2_memb_func, next_state[2])
            n_s3 = memb_func_eval(dr3_memb_func, next_state[3])

            actions_of_next_state = qtable[n_s0][n_s1][n_s2][n_s3]

            next_a0_idx = np.where(actions_of_next_state[0] == np.amax(actions_of_next_state[0]))[0][0]
            next_a1_idx = np.where(actions_of_next_state[1] == np.amax(actions_of_next_state[1]))[0][0]
            next_a2_idx = np.where(actions_of_next_state[2] == np.amax(actions_of_next_state[2]))[0][0]

            # TD Update:
            qtable[s0][s1][s2][s3][0][a0_idx] = actions_of_state[0][a0_idx] + alpha * (reward + gamma * actions_of_next_state[0][next_a0_idx] - actions_of_state[0][a0_idx])
            qtable[s0][s1][s2][s3][1][a1_idx] = actions_of_state[1][a1_idx] + alpha * (reward + gamma * actions_of_next_state[1][next_a1_idx] - actions_of_state[1][a1_idx])
            qtable[s0][s1][s2][s3][2][a2_idx] = actions_of_state[2][a2_idx] + alpha * (reward + gamma * actions_of_next_state[2][next_a2_idx] - actions_of_state[2][a2_idx])

            cum_reward += reward

        avg_r = cum_reward / steps

        print("... avg_reward: ", str(avg_r))
        print("")

        if e % 5 == 0:
            #np.save(r"\_saved\qtable", qtable)
            np.save("./_saved/qtable", qtable)

def online_learning(env, episode, step, gamma=0.9, alpha=0.5, epsilon=0.1, is_load = False):
    speed_map = [30, 40, 50, 60, 70, 80, 90, 100, 110]
    ufr_memb_func, dr1_memb_func, dr2_memb_func, dr3_memb_func, qtable = presetup_separated()

    if is_load:
        #qtable = np.load(r"\saved\qtable.npy")
        qtable = np.load("./_saved/qtable.npy")
        print("Q-Table is Loaded!")

    for e in range(episode):
        print(" ###########################: ", str(e))
        cum_reward = 0
        state = env.reset([110, 110, 110])

        for s in range(step):
            print(" ---------------------------: ", str(s))
            a0_idx, a1_idx, a2_idx = 0, 0, 0
            actions = []
            s0 = memb_func_eval(ufr_memb_func, state[0])
            s1 = memb_func_eval(dr1_memb_func, state[1])
            s2 = memb_func_eval(dr2_memb_func, state[2])
            s3 = memb_func_eval(dr3_memb_func, state[3])

            print(state)
            print(" @ state: ", str(s0), str(s1), str(s2), str(s3))
            actions_of_state = qtable[s0][s1][s2][s3]

            if random.uniform(0, 1) < epsilon:
                a0_idx = random.randrange(len(speed_map))
                a1_idx = random.randrange(len(speed_map))
                a2_idx = random.randrange(len(speed_map))
            else:
                a0_idx = np.where(actions_of_state[0] == np.amax(actions_of_state[0]))[0][0]
                a1_idx = np.where(actions_of_state[1] == np.amax(actions_of_state[1]))[0][0]
                a2_idx = np.where(actions_of_state[2] == np.amax(actions_of_state[2]))[0][0]
                
            actions = [speed_map[a0_idx], speed_map[a1_idx], speed_map[a2_idx]]
            print(actions)

            next_state, reward = env.step(actions)

            cum_reward += reward

            n_s0 = memb_func_eval(ufr_memb_func, next_state[0])
            n_s1 = memb_func_eval(dr1_memb_func, next_state[1])
            n_s2 = memb_func_eval(dr2_memb_func, next_state[2])
            n_s3 = memb_func_eval(dr3_memb_func, next_state[3])

            actions_of_next_state = qtable[n_s0][n_s1][n_s2][n_s3]

            next_a0_idx = np.where(actions_of_next_state[0] == np.amax(actions_of_next_state[0]))[0][0]
            next_a1_idx = np.where(actions_of_next_state[1] == np.amax(actions_of_next_state[1]))[0][0]
            next_a2_idx = np.where(actions_of_next_state[2] == np.amax(actions_of_next_state[2]))[0][0]

            # TD Update:
            qtable[s0][s1][s2][s3][0][a0_idx] = actions_of_state[0][a0_idx] + alpha * (reward + gamma * actions_of_next_state[0][next_a0_idx] - actions_of_state[0][a0_idx])
            qtable[s0][s1][s2][s3][1][a1_idx] = actions_of_state[1][a1_idx] + alpha * (reward + gamma * actions_of_next_state[1][next_a1_idx] - actions_of_state[1][a1_idx])
            qtable[s0][s1][s2][s3][2][a2_idx] = actions_of_state[2][a2_idx] + alpha * (reward + gamma * actions_of_next_state[2][next_a2_idx] - actions_of_state[2][a2_idx])

            state = next_state
            # garbage recycling
            gc.collect()

        avg_r = cum_reward / step
        #updateReport(r"\_report\episode_report.csv", [str(avg_r), str(e)])
        updateReport("/_report/episode_report.csv", [str(avg_r), str(e)])
            
        print("... avg_reward: ", str(avg_r))
        print("")

        if e % 5 == 0:
            #np.save(r"\_saved\qtable", qtable)
            np.save("./_saved/qtable", qtable)

def openai_learning(env, episode, step, gamma=0.9, alpha=0.5, epsilon=0.1, is_load = False):
    input_1_ms_dim, input_2_ms_dim, input_3_ms_dim, input_4_ms_dim = 50, 50, 50, 50
    output_ms_dim = 2
    output_dim = 1

    input_1_func = even_dist(-4.8, 4.8, input_1_ms_dim)
    input_2_func = even_dist(-3, 3, input_2_ms_dim)
    input_3_func = even_dist(-24, 24, input_3_ms_dim)
    input_4_func = even_dist(-3, 3, input_4_ms_dim)

    qtable = np.zeros((
        input_1_ms_dim,
        input_2_ms_dim,
        input_3_ms_dim,
        input_4_ms_dim,
        output_dim, 
        output_ms_dim)
    )

    if is_load:
        #qtable = np.load(r"\saved\qtable.npy")
        qtable = np.load("./_saved/qtable.npy")
        print("##############################")
        print("###   Q-Table is Loaded!   ###")
        print("##############################")

    for e in range(episode):
        print(" ###########################: ", str(e))
        step = 0
        state = env.reset()

        while True:
            step += 1
            #print(" ---------------------------: ", str(step))
            action = -1

            # Get current State
            s0 = memb_func_eval(input_1_func, state[0])
            s1 = memb_func_eval(input_2_func, state[1])
            s2 = memb_func_eval(input_3_func, state[2])
            s3 = memb_func_eval(input_4_func, state[3])

            #print(state)
            #print(" @ state: ", str(s0), str(s1), str(s2), str(s3))
            actions_of_state = qtable[s0][s1][s2][s3]

            if random.uniform(0, 1) < epsilon:
                action = random.choice([0, 1])
            else:
                action = np.where(actions_of_state[0] == np.amax(actions_of_state[0]))[0][0]
                
            #print(action)

            state_next, reward, terminal, info = env.step(action)

            if terminal:
                qtable[s0][s1][s2][s3][0][action] = -reward
                break
            else:
                n_s0 = memb_func_eval(input_1_func, state_next[0])
                n_s1 = memb_func_eval(input_2_func, state_next[1])
                n_s2 = memb_func_eval(input_3_func, state_next[2])
                n_s3 = memb_func_eval(input_4_func, state_next[3])

                actions_of_state_next = qtable[n_s0][n_s1][n_s2][n_s3]

                action_next = np.where(actions_of_state_next[0] == np.amax(actions_of_state_next[0]))[0][0]

                # TD Update:
                qtable[s0][s1][s2][s3][0][action] = actions_of_state[0][action] + alpha * (reward + gamma * actions_of_state_next[0][action_next] - actions_of_state[0][action])

                state = state_next

            

        #updateReport(r"\_report\episode_report.csv", [str(avg_r), str(e)])
        updateReport("/_report/episode_report.csv", [str(step), str(e)])
            
        print("... step: ", str(step))
        print("")

        if e % 5 == 0:
            #np.save(r"\_saved\qtable", qtable)
            np.save("./_saved/qtable", qtable)  

def openai_learning_replay(env, episode, step, gamma=0.9, alpha=0.5, epsilon=0.1, batch_size = 50, is_load = False):
    # ------------------------------------------------- hyperparameter setup
    curve = []
    best_score = 0
    memory = deque(maxlen=1000)
    input_1_ms_dim, input_2_ms_dim, input_3_ms_dim, input_4_ms_dim = 50, 50, 50, 50
    #input_1_ms_dim, input_2_ms_dim, input_3_ms_dim, input_4_ms_dim = 25, 25, 25, 25
    #input_1_ms_dim, input_2_ms_dim, input_3_ms_dim, input_4_ms_dim = 75, 75, 75, 75
    output_ms_dim = 2
    output_dim = 1

    # ------------------------------------------------- input membership functions
    input_1_func = even_dist(-4.8, 4.8, input_1_ms_dim)
    input_2_func = even_dist(-3, 3, input_2_ms_dim)
    input_3_func = even_dist(-24, 24, input_3_ms_dim)
    input_4_func = even_dist(-3, 3, input_4_ms_dim)

    # ------------------------------------------------- Q-Table / Rules
    qtable = np.zeros((
        input_1_ms_dim,
        input_2_ms_dim,
        input_3_ms_dim,
        input_4_ms_dim,
        output_dim, 
        output_ms_dim)
    )

    # ------------------------------------------------- Load pre-trained Q-Table
    if is_load:
        #qtable = np.load(r"\saved\qtable.npy")
        qtable = np.load("./_saved/qtable.npy")
        print("##############################")
        print("###   Q-Table is Loaded!   ###")
        print("##############################")

    for e in range(episode):
        step = 0
        state = env.reset()
        temp_memory = deque(maxlen=500)

        print(" ###########################: ", str(e))

        while True:
            step += 1
            action = 0

            # Get current State
            s0 = memb_func_eval(input_1_func, state[0])
            s1 = memb_func_eval(input_2_func, state[1])
            s2 = memb_func_eval(input_3_func, state[2])
            s3 = memb_func_eval(input_4_func, state[3])

            #print(state)
            #print(" @ state: ", str(s0), str(s1), str(s2), str(s3))
            actions_of_state = qtable[s0][s1][s2][s3]

            if random.uniform(0, 1) < epsilon:
                action = random.choice([0, 1])
            else:
                action = np.where(actions_of_state[0] == np.amax(actions_of_state[0]))[0][0]
                
            #print(action)

            state_next, reward, terminal, info = env.step(action)

            if terminal:
                if step >= best_score:
                    best_score = step
                    memory += temp_memory

                reward = -1
                qtable[s0][s1][s2][s3][0][action] = reward
                #qtable[s0][s1][s2][s3][0][action] = actions_of_state[0][action] + alpha * (reward - actions_of_state[0][action])
                break

            else:
                #reward *= (1 + step / 500)
                temp_memory.append((state, action, reward, state_next, terminal))

                n_s0 = memb_func_eval(input_1_func, state_next[0])
                n_s1 = memb_func_eval(input_2_func, state_next[1])
                n_s2 = memb_func_eval(input_3_func, state_next[2])
                n_s3 = memb_func_eval(input_4_func, state_next[3])

                actions_of_state_next = qtable[n_s0][n_s1][n_s2][n_s3]

                action_next = np.where(actions_of_state_next[0] == np.amax(actions_of_state_next[0]))[0][0]

                # TD Update:
                qtable[s0][s1][s2][s3][0][action] = actions_of_state[0][action] + alpha * (reward + gamma * actions_of_state_next[0][action_next] - actions_of_state[0][action])

                state = state_next

      
        #updateReport(r"\_report\episode_report.csv", [str(avg_r), str(e)])
        #updateReport("/_report/episode_report.csv", [str(step), str(e)])
        curve.append(step)
            

        """
        if step > best_score * 0.9:
            print("")
            print("")
            print(" =============== Update memory")
            print("")
            print("")
            memory += temp_memory
        """
 


        """
        # Batch Replay
        if len(memory) > batch_size:
            batch = random.sample(memory, batch_size)
            for _state, _action, _reward, _state_next, _terminal in batch:
                # Get current State
                _s0 = memb_func_eval(input_1_func, _state[0])
                _s1 = memb_func_eval(input_2_func, _state[1])
                _s2 = memb_func_eval(input_3_func, _state[2])
                _s3 = memb_func_eval(input_4_func, _state[3])

                _n_s0 = memb_func_eval(input_1_func, _state_next[0])
                _n_s1 = memb_func_eval(input_2_func, _state_next[1])
                _n_s2 = memb_func_eval(input_3_func, _state_next[2])
                _n_s3 = memb_func_eval(input_4_func, _state_next[3])

                _actions_of_state_next = qtable[_n_s0][_n_s1][_n_s2][_n_s3]
                _action_next = np.where(_actions_of_state_next[0] == np.amax(_actions_of_state_next[0]))[0][0]

                # TD Update:
                q_old = qtable[_s0][_s1][_s2][_s3][0][_action]
                q_new = qtable[_n_s0][_n_s1][_n_s2][_n_s3][0][_action_next]

                qtable[_s0][_s1][_s2][_s3][0][_action] = q_old + alpha * (_reward + gamma * q_new - q_old)
                #qtable[s0][s1][s2][s3][0][_action] = (1 - alpha) * q_old + alpha * (_reward)
        """


        if e % 5 == 0:
            # Batch Replay
            if len(memory) > batch_size:
                batch = random.sample(memory, batch_size)
                for _state, _action, _reward, _state_next, _terminal in batch:
                    # Get current State
                    _s0 = memb_func_eval(input_1_func, _state[0])
                    _s1 = memb_func_eval(input_2_func, _state[1])
                    _s2 = memb_func_eval(input_3_func, _state[2])
                    _s3 = memb_func_eval(input_4_func, _state[3])

                    _n_s0 = memb_func_eval(input_1_func, _state_next[0])
                    _n_s1 = memb_func_eval(input_2_func, _state_next[1])
                    _n_s2 = memb_func_eval(input_3_func, _state_next[2])
                    _n_s3 = memb_func_eval(input_4_func, _state_next[3])

                    _actions_of_state_next = qtable[_n_s0][_n_s1][_n_s2][_n_s3]
                    _action_next = np.where(_actions_of_state_next[0] == np.amax(_actions_of_state_next[0]))[0][0]

                    # TD Update:
                    q_old = qtable[_s0][_s1][_s2][_s3][0][_action]
                    q_new = qtable[_n_s0][_n_s1][_n_s2][_n_s3][0][_action_next]

                    qtable[_s0][_s1][_s2][_s3][0][_action] = q_old + alpha * (_reward + gamma * q_new - q_old)
                    #qtable[s0][s1][s2][s3][0][_action] = (1 - alpha) * q_old + alpha * (_reward)

            #np.save(r"\_saved\qtable", qtable)
            plt.plot(curve)
            plt.xlabel('x - iteration')
            plt.ylabel('y - score')
            plt.title('performance graph')
            plt.savefig('./_plot/plot.png')
            #np.save("./_saved/qtable", qtable)  


"""
def get_map(dim_shape):
    count = 0
    speed_map = [40, 50, 60, 70, 80, 90, 100, 110, 120]
    p_map = {}
    speed_map = {}

    for a in range(dim_shape[0]):
        for b in range(dim_shape[1]):
            for c in range(dim_shape[2]):
                for d in range(dim_shape[3]):
                    p_map[str(a)+str(b)+str(c)+str(d)] = count
                    count += 1

    return p_map, speed_map

def presetup_one():
    return False
"""

if __name__ == '__main__':
    """
    conda deactivate
    """
    ENV_NAME = "CartPole-v1"
    episode = 9999#999999
    step = 50
    gamma = 0.9
    alpha = 0.1
    epsilon = 0.1
    batch_size = 500
    is_load = False
    #env = Env_Test()

    env = gym.make(ENV_NAME)

    #openai_learning(env, episode, step, gamma, alpha, epsilon, is_load)
    openai_learning_replay(env, episode, step, gamma, alpha, epsilon, batch_size, is_load)
    #online_learning(env, episode, step, gamma, alpha, epsilon, is_load)
    """
    offline_learning(1, step, gamma, alpha, epsilon, is_load)
    """


