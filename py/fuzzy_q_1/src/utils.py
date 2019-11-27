"""
Author: XIN LI
"""

import os
import csv
import numpy as np

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
	x = []
	y = []

	for i in range(step):
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

