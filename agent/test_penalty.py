import numpy as np

state1 = np.zeros((4, 15, 15))
state1[2][5][5] = 2.0 # bot head
state1[2][5][6] = 1.0 # bot tail

state2 = np.zeros((4, 15, 15))
state2[2][5][5] = 2.0
state2[2][5][6] = 1.0

# Simulate penalty script
if np.array_equal(state1[2], state2[2]):
    print("PENALIZED")
