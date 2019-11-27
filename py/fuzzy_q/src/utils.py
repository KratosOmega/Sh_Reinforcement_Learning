"""
Author: XIN LI
"""

import os
import csv
import numpy as np
import itertools
import operator


def load_data(file_name):
	dataset = []
	path = os.getcwd() + file_name
	with open(path) as fileReader:
	    csv_reader = csv.reader(fileReader, delimiter=',')

	    for record in csv_reader:

	    	s = list(map(float, record[0].split(";")))
	    	a = list(map(float, record[1].split(";")))
	    	r = float(record[2])
	    	s_ = list(map(float, record[3].split(";")))

	    	dataset.append([s, a, r, s_])

	fileReader.close()

	return dataset


def updateReport(file_name, entry):
	path = os.getcwd() + file_name
	with open(path, 'a', newline='') as fileWriter:
		writer = csv.writer(fileWriter)
		writer.writerow(entry)
	fileWriter.close()

def even_dist(bound_l, bound_r, num_members):
	cur_ctr = bound_l
	domain = bound_r - bound_l
	step = domain / (num_members - 1)

	foot = []

	for m in range(num_members):
		foot.append([cur_ctr - step, cur_ctr, cur_ctr + step])
		cur_ctr = cur_ctr + step

	return np.array(foot)

def memb_func_eval(memb_func, val):
	eval_max = 0
	output = -1
	step = memb_func.shape[0]

	for i in range(step):
		x, y = [], []
		if memb_func[i][0] <= val and val <memb_func[i][2]:
			if memb_func[i][0] <= val and val <memb_func[i][1]:
				x.append(memb_func[i][0])
				x.append(memb_func[i][1])
				y.append(0)
				y.append(1)
			if memb_func[i][1] <= val and val <memb_func[i][2]:
				x.append(memb_func[i][1])
				x.append(memb_func[i][2])
				y.append(1)
				y.append(0)
			coefficients = np.polyfit(x, y, 1)
			poly = np.poly1d(coefficients)
			val_eval = poly(val)
			if val_eval > eval_max:
				eval_max = val_eval
				output = i

	return output

# group membership function eval
def state_eval(memb_func_list, val_list):
	state_list = []
	state_prob_list = []
	res_state_list = []
	res_state_prob_list = []

	for i in range(len(val_list)):
		val = val_list[i]
		memb_func = memb_func_list[i]

		state_temp_list = []
		state_temp_prob_list = []
		step = memb_func.shape[0]

		for i in range(step):
			x, y = [], []
			if memb_func[i][0] <= val and val <memb_func[i][2]:
				if memb_func[i][0] <= val and val <memb_func[i][1]:
					x.append(memb_func[i][0])
					x.append(memb_func[i][1])
					y.append(0)
					y.append(1)
				if memb_func[i][1] <= val and val <memb_func[i][2]:
					x.append(memb_func[i][1])
					x.append(memb_func[i][2])
					y.append(1)
					y.append(0)
				coefficients = np.polyfit(x, y, 1)
				poly = np.poly1d(coefficients)
				val_eval = poly(val)
				state_temp_list.append(i)
				state_temp_prob_list.append(val_eval)
		state_list.append(state_temp_list)
		state_prob_list.append(state_temp_prob_list)

	state_perm = list(itertools.product(*state_list))
	state_prob_perm = list(itertools.product(*state_prob_list))

	state_prob = []
	for i in state_prob_perm:
		state_prob.append(np.product(i))

	for i in range(len(state_prob)):
		s = state_perm[i]
		p = state_prob[i]

		if p >= 0.01:
			#res_state_list.append(list(s))
			res_state_list.append(s)
			res_state_prob_list.append(p)

	""" TODO: probabilities are already normalized during permutation product?
	#------------------------------------ normalize probs
	probs_sum = sum(res_state_prob_list)

	for i in range(len(res_state_prob_list)):
		res_state_prob_list[i] = res_state_prob_list[i] / probs_sum

	#------------------------------------
	"""

	return res_state_list, res_state_prob_list


def vote_council(state_list, state_prob_list, q_table):
	vote_record = []
	vote_pool = {
		0 : 0,
		1 : 0,
	}

	for i in range(len(state_list)):
		state = state_list[i]
		state_prob = state_prob_list[i]

		actions_of_state = q_table[state]

		"""
		print(" -------- ")
		print(actions_of_state)
		print(" -------- ")
		"""

		action = np.where(actions_of_state[0] == np.amax(actions_of_state[0]))[0][0]

		vote_pool[action] += state_prob
		action_participate = state_prob
		vote_record.append([state, state_prob, action, action_participate])

	action = max(vote_pool.items(), key = operator.itemgetter(1))[0] 

	for i in range(len(vote_record)):
		action_taked = vote_record[i][2]
		vote_record[i][3] = vote_record[i][1] / vote_pool[action_taked]

	return action, vote_record



def update_council(alpha, gamma, reward, terminal, qtable, action, vote_record, vote_next_record = []):
	action_taked_record = []

	for r in vote_record:
		if r[2] == action:
			action_taked_record.append(r)

	if terminal:
		reward *= -1

		for r in action_taked_record:
			# r[0] is the state idex
			# r[1] is the state probability
			# r[2] is the action picked
			q_old = qtable[r[0]][0][action]

			weight =  r[3] * alpha

			#qtable[r[0]][0][action] = (1 - weight) * q_old + weight * reward
			qtable[r[0]][0][action] = (1 - weight) * q_old + weight * reward
	else:
		# calculate accumulative next state value
		q_new = 0

		for r in vote_next_record:
			q_new +=  r[3] * qtable[r[0]][0][r[2]]

		for r in action_taked_record:
			# r[0] is the state idex
			# r[1] is the state probability
			# r[2] is the action picked
			q_old = qtable[r[0]][0][action]

			weight =  r[3] * alpha

			qtable[r[0]][0][action] = (1 - weight) * q_old + weight * (reward + gamma * q_new)

	return qtable
                


                
















