import numpy as np

LR = 0.0005
EPSILON = 0.6
MEMORY_CAPACITY = 2000
count_episode = 0
test_num = 10
BATCH_SIZE = 256
GAMMA = 0.99

# iteration numbers during each output
ite_num = 500

# Output Filename
model_file = 'save/RL.pkl'
test_file = 'save/test_result.txt'
episode_record = 'save/episode_record.txt'
epsilon_record = 'save/epsilon_record.txt'

original_vec = np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 1])
empty_vec = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
