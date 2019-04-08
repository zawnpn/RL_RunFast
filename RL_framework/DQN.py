import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# utils
from extra.utils import trans_vector, get_cards_small_extend, calculate_score

# config
from extra.config import original_vec, LR, MEMORY_CAPACITY, BATCH_SIZE, GAMMA


class QNet(nn.Module):
    def __init__(self, ):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(209, 512)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.fc2 = nn.Linear(512, 256)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.fc3 = nn.Linear(256, 128)
        self.fc3.weight.data.normal_(0, 0.1)  # initialization
        # self.fc4 = nn.Linear(160, 150)
        # self.fc4.weight.data.normal_(0, 0.1)  # initialization
        # self.fc5 = nn.Linear(150, 120)
        # self.fc5.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(128, 1)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
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
        self.eval_net, self.target_net = QNet(), QNet()
        self.eval_net.cuda()
        self.target_net.cuda()
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.memory_counter = 0  # for storing memory
        self.memory_counter_ = 0  # for storing memory
        self.memory = np.zeros((self.MEMORY_CAPACITY, 277))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.loss_func.cuda()

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
        # if len(b_memory) < BATCH_SIZE:
        #     return
        # get s,a,r,s_ according to positions

        b_s = torch.FloatTensor(b_memory[:, 1+143:1+143+66])
        b_s_ = torch.FloatTensor(b_memory[:, 1+143+66:1+143+66+66])
        b_a = torch.FloatTensor(b_memory[:, 1:1+143])
        b_r = torch.FloatTensor(b_memory[:, -1])

        input = torch.cat((b_a, b_s), 1).cuda()
        input_ = torch.cat((b_a, b_s_), 1).cuda()

        # double Q-learning for TD methods setting, useless in MC methods
        # q_eval w.r.t the action in experience
        # q_eval = self.eval_net(b_s).gather(1.0, b_a)  # shape (batch, 1)
        #
        q_eval = self.eval_net(input).squeeze()  # shape (batch, 1)
        q_next = self.target_net(input_).squeeze()
        # double Q-learning for TD methods setting, useless in MC methods
        # q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        # q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        #
        # q_target = b_r.cuda()
        q_target = b_r.cuda() + GAMMA * q_next
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), q_eval.data.cpu().numpy().mean()

    def choose_action(self, player, EPSILON, ways_toplay=[]):
        status = player.status
        if status == np.array([0]):
            pattern_to_playcards = [player.current_pattern]
        else:
            pattern_to_playcards = player.search_pattern()

        biggest_w = 0
        biggest_p = 0
        biggest_temp = 0
        ways_list = []
        lp = len(pattern_to_playcards)
        for p in range(lp):
            if status == np.array([1]):
                ways_toplay = player.search_play_methods(pattern_to_playcards[p])
                if len(ways_toplay) == 0:
                    return []
            ways_list.append(ways_toplay)

            lw = len(ways_toplay)
            for w in range(lw):
                action_small_extend = get_cards_small_extend(ways_toplay[w], pattern_to_playcards[p])
                player_state = player.get_state()
                input = np.concatenate((action_small_extend, player_state), axis=0)
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
        action_cards = ways_list[biggest_p][biggest_w]
        player.current_pattern = pattern_to_playcards[biggest_p]
        # action_Q = biggest_temp
        if type(action_cards) == int:
            action_cards = [action_cards]
        return action_cards

    # 每次出牌结束，存储当时的局面s和出牌a, 和 ABC的位置, 每局结束时, 根据 ABC 的位置去分配奖励reward
    def store_transition(self, player, action_cards, current_state):
        pattern = player.current_pattern
        cards_small_extend = get_cards_small_extend(action_cards, pattern)
        current_position = player.position
        if current_position == 'player_A':
            position = 1
        if current_position == 'player_B':
            position = 2
        if current_position == 'player_C':
            position = 3
        position = np.array([position])
        next_state = player.get_state()
        r_placeholder = np.array([0])
        input = np.concatenate((position,cards_small_extend,current_state,next_state,r_placeholder),axis=0)
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = input
        self.memory_counter += 1

    # 每局游戏结束，为训练标签添加奖励
    def add_reward(self, player, player_A, player_B, player_C):

        # get info from current player
        winner = player.position
        next_player = player.get_next_player()
        next_next_player = next_player.get_next_player()
        loser_one = next_player.position
        loser_two = next_next_player.position
        loser_one_cards = next_player.cards
        loser_two_cards = next_next_player.cards

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

            #赢的人记分，输的人剩余多少牌输多少分
            winner_score = 1
            loser_one_cards_small = trans_vector(loser_one_cards)
            loser_two_cards_small = trans_vector(loser_two_cards)
            loser_one_score = calculate_score(loser_one_cards_small)
            loser_two_score = calculate_score(loser_two_cards_small)

            # 将炸弹的分与剩牌失去分共同作为当局reward
            # if self.memory[temp, 0] == winner_position:
            #     self.memory[temp, -1] = (winner_score + winner_boom_success*20 - loser_one_boom_success*10 - loser_two_boom_success*10)
            # elif self.memory[temp, 0] == loser_one_position:
            #     self.memory[temp, -1] = (loser_one_score + loser_one_boom_success*20 - loser_two_boom_success*10 - winner_boom_success*10)
            # elif self.memory[temp, 0] == loser_two_position:
            #     self.memory[temp, -1] = (loser_two_score + loser_two_boom_success*20 - winner_boom_success*10 - loser_one_boom_success*10)
            if self.memory[temp, 0] == winner_position:
                self.memory[temp, -1] = winner_score
            elif self.memory[temp, 0] == loser_one_position:
                self.memory[temp, -1] = loser_one_score
            elif self.memory[temp, 0] == loser_two_position:
                self.memory[temp, -1] = loser_two_score
        self.memory_counter_ = self.memory_counter