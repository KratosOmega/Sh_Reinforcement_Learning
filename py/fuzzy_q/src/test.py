from utils import updateReport, load_data, even_dist, memb_func_eval, state_eval, vote_council
import numpy as np

"""
a = even_dist(0, 10, 5)
b = even_dist(0, 5, 5)
c = even_dist(0, 10, 2)

v1 = 0
v2 = 2
v3 = 8

t1, t2 = state_eval([a, b, c], [v1, v2, v3])
print(t1)
print(t2)

qtable = np.random.rand(3,3,3,1,2)


print(qtable[(1,1,1)])

action, vote_record = vote_council(t1, t2, qtable)

print(action)
print(vote_record)
"""


qtable = np.random.rand(
        2,
        2,
        2,
        2,
        1, 
        2
)

qtable_0 = np.zeros((
        2,
        2,
        2,
        2,
        1, 
        2)
)

print(qtable)
print(qtable*0.01)
print(qtable_0.shape)

