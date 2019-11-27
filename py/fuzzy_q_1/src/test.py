import numpy as np
import random
from collections import deque
import operator

input_1_ms_dim, input_2_ms_dim, input_3_ms_dim, input_4_ms_dim = 50, 50, 50, 50
output_ms_dim = 2
output_dim = 1

qtable = np.zeros((
	input_1_ms_dim,
	input_2_ms_dim,
	input_3_ms_dim,
	input_4_ms_dim,
	output_dim, 
	output_ms_dim)
)

qtable[0][0][0][0][0][0] = 0.6
qtable[0][0][0][0][0][1] = 0.9
print(qtable[0][0][0][0][0])

x = np.where(qtable[0][0][0][0][0] == np.amax(qtable[0][0][0][0][0]))[0][0]

print(x)