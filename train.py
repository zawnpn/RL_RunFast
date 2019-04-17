import os
import torch
import numpy as np
from extra.utils import divide_cards, trans_vector
from GameEnv.RunFastGame import RunfastGameEnv
from RL_framework.DQN import DQN, QNet
from extra.config import model_A_file, model_B_file, model_C_file, EPSILON, const_EPSILON, count_episode, episode_record, epsilon_record
from tensorboardX import SummaryWriter


if __name__ == '__main__':
    if os.path.exists(model_A_file):
        RLA = torch.load(model_A_file)
        print('Detected model A file, load DQN from it.\n')
    else:
        RLA = DQN()
        print('No model A file, create a new DQN model.\n')
    if os.path.exists(model_B_file):
        RLB = torch.load(model_B_file)
        print('Detected model B file, load DQN from it.\n')
    else:
        RLB = DQN()
        print('No model B file, create a new DQN model.\n')
    if os.path.exists(model_C_file):
        RLC = torch.load(model_C_file)
        print('Detected model C file, load DQN from it.\n')
    else:
        RLC = DQN()
        print('No model C file, create a new DQN model.\n')
    if os.path.exists(episode_record):
        with open(episode_record, 'r') as f:
            count_episode = int(f.read())
    if os.path.exists(epsilon_record):
        with open(epsilon_record, 'r') as f:
            EPSILON = float(f.read())
    player_A = RunfastGameEnv()
    player_B = RunfastGameEnv()
    player_C = RunfastGameEnv()
    player_A.update_next(player_B)
    player_B.update_next(player_C)
    player_C.update_next(player_A)
    player_A.update_position('player_A')
    player_B.update_position('player_B')
    player_C.update_position('player_C')
    writer = SummaryWriter()
    dummy_input = torch.rand(1, 209)
    test_QNet = QNet()
    with SummaryWriter(comment='QNet') as w:
        w.add_graph(test_QNet, (dummy_input,))
    start_game = 'y'
    if start_game == 'y' or start_game == 'Y' or start_game == 'YES' or start_game == 'yes' or start_game == 'Yes':
        is_playing = True
    else:
        is_playing = False
    count_epoch = 0
    Q_A, Q_B, Q_C = 0, 0, 0
    while(is_playing):
        a, b, c = divide_cards()
        player_A.cards_used  = np.zeros(13)
        player_B.cards_used = np.zeros(13)
        player_C.cards_used = np.zeros(13)
        player_A.set_cards(a)
        player_B.set_cards(b)
        player_C.set_cards(c)
        #选择初始玩家，黑桃三出头
        player_A.boom_success = 0
        player_B.boom_success = 0
        player_C.boom_success = 0

        if 0 in a:
            current_player = player_A
            privious_player = player_A
        elif 0 in b:
            current_player = player_B
            privious_player = player_B
        else:
            current_player = player_C
            privious_player = player_C
        isEnd = False

        current_pattern = 0
        biggestcards = []
        current_state = current_player.get_state()
        while(isEnd==False ):
            if current_player == privious_player:
                current_player.status = np.array([1])
                # record who play boom（炸弹） and boom_success+1
                if len(biggestcards) == 4:
                    if int(biggestcards[0] / 4) == int(biggestcards[3] / 4) and (biggestcards[0] / 4 < 11):
                        current_player.boom_success = current_player.boom_success + 1

                biggestcards = []

                # choose Action from State.
                if current_player.position == 'player_A':
                    action_cards = RLA.choose_action(current_player, EPSILON)
                elif current_player.position == 'player_B':
                    action_cards = RLB.choose_action(current_player, np.random.uniform(EPSILON / 3, EPSILON / 2))
                elif current_player.position == 'player_C':
                    action_cards = RLC.choose_action(current_player, const_EPSILON)

                # implement Action
                small_cards = trans_vector(action_cards)
                current_player.cards_used = current_player.cards_used + small_cards
                biggestcards = action_cards
                if len(biggestcards)==4 and int(biggestcards[0]/4)==int(biggestcards[3]/4):
                    if int(biggestcards[0]/4)==int(biggestcards[3]/4) and (biggestcards[0]/4<11):
                        current_player.current_pattern = 3
                if current_player.position == 'player_A':
                    RLA.store_transition(current_player, action_cards, current_state)
                elif current_player.position == 'player_B':
                    RLB.store_transition(current_player, action_cards, current_state)
                elif current_player.position == 'player_C':
                    RLC.store_transition(current_player, action_cards, current_state)
                current_state = current_player.get_state()
                isEnd = current_player.play_cards(action_cards)
                current_pattern = current_player.current_pattern
                ## Q-learning, calculate reward and add reward
                if isEnd:
                    # record that if the cards played lastest are boom
                    if len(biggestcards) == 4:
                        if int(biggestcards[0] / 4) == int(biggestcards[3] / 4) and (biggestcards[0] / 4 < 11):
                            current_player.boom_success = current_player.boom_success + 1
                    RLA.add_reward(current_player)
                    RLB.add_reward(current_player)
                    RLC.add_reward(current_player)
                ## Q-learning, calculate reward and add reward
                if isEnd:
                    count_episode = count_episode+1
                    break
                else:
                    current_player = current_player.get_next_player()

            else:
                current_player.current_pattern = current_pattern
                ways_toplay = current_player.search_play_methods(current_pattern, biggestcards)
                if len(ways_toplay)==0:
                    current_player = current_player.get_next_player()
                else:
                    current_player.status = np.array([0])
                    # choose Action from State.
                    if current_player.position == 'player_A':
                        action_cards = RLA.choose_action(current_player, EPSILON, ways_toplay=ways_toplay)
                    elif current_player.position == 'player_B':
                        action_cards = RLB.choose_action(current_player, np.random.uniform(EPSILON / 3, EPSILON / 2), ways_toplay=ways_toplay)
                    elif current_player.position == 'player_C':
                        action_cards = RLC.choose_action(current_player, const_EPSILON, ways_toplay=ways_toplay)
                    small_cards = trans_vector(action_cards)
                    current_player.cards_used = current_player.cards_used + small_cards

                    biggestcards = action_cards
                    if len(biggestcards) == 4:
                        if int(biggestcards[0] / 4) == int(biggestcards[3] / 4) and (biggestcards[0] / 4 < 11):
                            current_pattern = 3
                    privious_player = current_player
                    if current_player.position == 'player_A':
                        RLA.store_transition(current_player, action_cards, current_state)
                    elif current_player.position == 'player_B':
                        RLB.store_transition(current_player, action_cards, current_state)
                    elif current_player.position == 'player_C':
                        RLC.store_transition(current_player, action_cards, current_state)
                    current_state = current_player.get_state()
                    isEnd = current_player.play_cards(action_cards)
                    # current_pattern = current_player.current_pattern
                    ## Q-learning, calculate reward and add reward,winner reward = 1,loser = 0
                    if isEnd:
                        if len(biggestcards) == 4:
                            if int(biggestcards[0] / 4) == int(biggestcards[3] / 4) and (biggestcards[0] / 4 < 11):
                                current_player.boom_success = current_player.boom_success + 1
                        RLA.add_reward(current_player)
                        RLB.add_reward(current_player)
                        RLC.add_reward(current_player)
                    if isEnd:
                        count_episode = count_episode+1
                        break

                    else:
                        current_player = current_player.get_next_player()

        if RLA.memory_counter % RLA.MEMORY_CAPACITY == 0:
            count_epoch += 1
            loss_value_a, q_value_a = RLA.learn()
            loss_value_b, q_value_b = RLB.learn()
            loss_value_c, q_value_c = RLC.learn()
            Q_A += q_value_a
            Q_B += q_value_b
            Q_C += q_value_c
            writer.add_scalars('train_loss', {
                'Loss_A': loss_value_a,
                'Loss_B': loss_value_b,
                'Loss_C': loss_value_c,
            }, count_epoch)
            writer.add_scalars('Current_Q_value', {
                'cur_Q_A': q_value_a,
                'cur_Q_B': q_value_b,
                'cur_Q_C': q_value_c,
            }, count_epoch)
            writer.add_scalars('Accumulate_Q_value', {
                'Q_A': Q_A,
                'Q_B': Q_B,
                'Q_C': Q_C,
            }, count_epoch)
            # if count_episode % ite_num == 0 and count_episode != 0:
            torch.save(RLA, model_A_file)
            torch.save(RLB, model_B_file)
            torch.save(RLC, model_C_file)
            print("Episode: %s; Loss: %s; Q: %s; eps: %s" % (count_episode, loss_value_a, q_value_a, EPSILON))
            with open(episode_record, 'w') as f:
                f.write(str(count_episode))
            with open(epsilon_record, 'w') as f:
                f.write(str(EPSILON))


        # print('是否继续游戏，请输入y或者n：')
        # if count_episode <= 1000000:
        #     start_game = 'y'
        if count_episode % 10000 == 0 and EPSILON < 0.99:
            EPSILON = EPSILON + 0.01