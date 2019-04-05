import numpy as np

LR = 0.01
EPSILON = 0.6
MEMORY_CAPACITY = 2000
count_episode = 0
test_num = 10
BATCH_SIZE = 512
GAMMA = 0.999

# iteration numbers during each output
ite_num = 500

# Output Filename
model_file = 'save/RL.pkl'
test_file = 'save/test_result.txt'
epoch_record = 'save/epoch_record.txt'

original_vec = np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 1])
empty_vec = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

pattern_len = {
    0: 1,
    1: 2,
    2: 4,  # 长度本应为 3 ，但可能有三带二，故暂定为 4
    3: 4,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
    10: 11,
    11: 12,
    12: 4,
    13: 6,
    14: 8,
    15: 10,
    16: 12,
    17: 14,
    18: 16,
    19: 6,
    20: 9,
    21: 12
}
