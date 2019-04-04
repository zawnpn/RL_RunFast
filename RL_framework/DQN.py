import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# utils
from extra.utils import trans_vector, get_cards_small_extend, calculate_score

# config
from extra.config import original_vec, LR, EPSILON, MEMORY_CAPACITY, BATCH_SIZE


class QNet(nn.Module):
    def __init__(self, ):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(209, 196)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.fc2 = nn.Linear(196, 179)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        # self.fc3 = nn.Linear(179, 160)
        # self.fc3.weight.data.normal_(0, 0.1)  # initialization
        # self.fc4 = nn.Linear(160, 150)
        # self.fc4.weight.data.normal_(0, 0.1)  # initialization
        # self.fc5 = nn.Linear(150, 120)
        # self.fc5.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(179, 1)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        # x = self.fc3(x)
        # x = F.relu(x)
        # x = self.fc4(x)
        # x = F.relu(x)
        # x = self.fc5(x)
        # x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        # double Q-learning for TD methods setting, useless in MC methods
        #     self.eval_net, self.target_net = Net(), Net()
        #     self.learn_step_counter = 0  # for target updating
        #
        self.eval_net = QNet()
        self.eval_net.cuda()
        self.MEMORY_CAPACITY = 2000
        self.memory_counter = 0  # for storing memory
        self.memory_counter_ = 0  # for storing memory
        self.memory = np.zeros((self.MEMORY_CAPACITY, 211))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.loss_func.cuda()

    def choose_action(self, player, ting, pattern_to_playcards, follow=False, ways_toplay=[]):
        current_player_cards = player.cards
        current_player_cards_used = player.cards_used
        next_player = player.get_next_player()
        next_next_player = next_player.get_next_player()
        next_player_cards_used = next_player.cards_used
        next_next_player_cards_used = next_next_player.cards_used
        biggest_w = 0
        biggest_p = 0
        biggest_temp = 0
        ways_list = []
        lp = len(pattern_to_playcards)
        for p in range(lp):
            if not follow:
                ways_toplay = player.search_play_methods(pattern_to_playcards[p])
            ways_list.append(ways_toplay)
            if len(ways_toplay) == 0:
                return []
            lw = len(ways_toplay)
            for w in range(lw):
                way_to_play_small_extend = get_cards_small_extend(ways_toplay[w], pattern_to_playcards[p])
                current_player_cards_small = trans_vector(current_player_cards)
                cards_left = original_vec - current_player_cards_used - next_player_cards_used - next_next_player_cards_used - current_player_cards_small
                input = np.concatenate((way_to_play_small_extend, current_player_cards_small,
                                        current_player_cards_used, next_player_cards_used,
                                        next_next_player_cards_used, cards_left, ting), axis=0)
                input = torch.from_numpy(input).float().cuda()
                temp = self.eval_net(input)
                if temp >= biggest_temp:
                    biggest_w = w
                    biggest_p = p
                    biggest_temp = temp
        if np.random.uniform() > EPSILON:  # greedy
            biggest_p = np.random.randint(0, lp)
            lw = len(ways_list[biggest_p])
            biggest_w = np.random.randint(0, lw)
        output_cards = ways_list[biggest_p][biggest_w]
        output_pattern = pattern_to_playcards[biggest_p]
        output_Q = biggest_temp
        if type(output_cards) == int:
            output_cards = [output_cards]
        return output_cards, output_pattern, output_Q

    def learn(self):
        # double Q-learning for TD methods setting, useless in MC methods
        # target parameter update
        # if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
        #     self.target_net.load_state_dict(self.eval_net.state_dict())
        # self.learn_step_counter += 1
        #

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        # get s,a,r,s_ according to positions

        b_s = torch.FloatTensor(b_memory[:, 144:-1])
        b_a = torch.FloatTensor(b_memory[:, 1:144])
        b_r = torch.FloatTensor(b_memory[:, -1])

        input = torch.cat((b_a, b_s), 1).cuda()

        # double Q-learning for TD methods setting, useless in MC methods
        # q_eval w.r.t the action in experience
        # q_eval = self.eval_net(b_s).gather(1.0, b_a)  # shape (batch, 1)
        #

        q_eval = self.eval_net(input).squeeze()  # shape (batch, 1)
        # double Q-learning for TD methods setting, useless in MC methods
        # q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        # q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        #
        q_target = b_r.cuda()
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # 每次出牌结束，存储当时的局面s和出牌a, 和 ABC的位置, 每局结束时, 根据 ABC 的位置去分配奖励reward
    def store_transition(self, cards, cards_inhand, cards_used, next_cards_used, next_next_cards_used, current_position, ting, pattern):
        original_vec = np.array([4,4,4,4,4,4,4,4,4,4,4,3,1])
        cards_small = trans_vector(cards)
        cards_left = original_vec - cards_used - next_cards_used - next_next_cards_used - cards_small
        if current_position == 'player_A':
            position = 1
        if current_position == 'player_B':
            position = 2
        if current_position == 'player_C':
            position = 3
        cards_small_extend = get_cards_small_extend(cards, pattern)
        cards_inhand_small = trans_vector(cards_inhand)
        position = np.array([position])
        flag = np.array([0])
        input = np.concatenate((position, cards_small_extend, cards_inhand_small, cards_used, next_cards_used, next_next_cards_used, cards_left, ting, flag), axis=0)
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = input
        self.memory_counter += 1

    # 每局游戏结束，为训练标签添加奖励
    def add_reward(self, winner, loser_one, loser_two, loser_one_cards, loser_two_cards, player_A, player_B, player_C):

        # boom_success 表示成功出炸弹次数，根据规则每次成功出炸弹，出牌者奖励为20，另外两人奖励为-10
        if winner == 'player_A':
            winner_position = 1
            winner_boom_success = player_A.boom_success
        elif winner == 'player_B':
            winner_position = 2
            winner_boom_success = player_B.boom_success
        elif winner == 'player_C':
            winner_position = 3
            winner_boom_success = player_C.boom_success

        if loser_one == 'player_A':
            loser_one_position = 1
            loser_one_boom_success = player_A.boom_success
        elif loser_one == 'player_B':
            loser_one_position = 2
            loser_one_boom_success = player_B.boom_success
        elif loser_one == 'player_C':
            loser_one_position = 3
            loser_one_boom_success = player_C.boom_success

        if loser_two == 'player_A':
            loser_two_position = 1
            loser_two_boom_success = player_A.boom_success
        elif loser_two == 'player_B':
            loser_two_position = 2
            loser_two_boom_success = player_B.boom_success
        elif loser_two == 'player_C':
            loser_two_position = 3
            loser_two_boom_success = player_C.boom_success

        for i in range(self.memory_counter - self.memory_counter_):
            index = self.memory_counter_ % MEMORY_CAPACITY
            temp = (index + i) % MEMORY_CAPACITY

            #赢的人记0分，输的人剩余多少牌输多少分
            winner_score = 0
            loser_one_cards_small = trans_vector(loser_one_cards)
            loser_two_cards_small = trans_vector(loser_two_cards)
            loser_one_score = calculate_score(loser_one_cards_small)
            loser_two_score = calculate_score(loser_two_cards_small)

            # 将炸弹的分与剩牌失去分共同作为当局reward
            if self.memory[temp, 0] == winner_position:
                self.memory[temp, -1] = (winner_score + winner_boom_success*20 - loser_one_boom_success*10 - loser_two_boom_success*10)
            elif self.memory[temp, 0] == loser_one_position:
                self.memory[temp, -1] = (loser_one_score + loser_one_boom_success*20 - loser_two_boom_success*10 - winner_boom_success*10)
            elif self.memory[temp, 0] == loser_two_position:
                self.memory[temp, -1] = (loser_two_score + loser_two_boom_success*20 - winner_boom_success*10 - loser_one_boom_success*10)
        self.memory_counter_ = self.memory_counter