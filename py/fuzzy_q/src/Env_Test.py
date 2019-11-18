import random
import numpy as np 

class Env_Test(object):
	def __init__(self):
		self.state_space = np.ndarray(shape=(6,), dtype=float)
		self.action_space = np.ndarray(shape=(3,), dtype=float)

	def reset(self):
		return [1000, 1000, 1000, 1000, 1000, 1000]

	def step(self, a):
		return [2000, 2000, 2000, 2000, 2000, 2000], 100, False