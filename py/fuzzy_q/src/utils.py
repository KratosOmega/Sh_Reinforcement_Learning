"""
Author: XIN LI
"""

import os
import csv

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

	return foot

