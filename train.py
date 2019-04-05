import os
import torch
import numpy as np
from extra.utils import divide_cards, trans_vector
from GameEnv.RunFastGame import RunfastGameEnv
from RL_framework.DQN import DQN
from extra.config import ite_num, model_file, EPSILON, count_episode, epoch_record

if __name__ == '__main__':
    if os.path.exists(model_file):
        RL = torch.load(model_file)
        print('Detected model file, load DQN from it.\n')
    else:
        RL = DQN()
        print('No model file, create a new DQN model.\n')
    if os.path.exists(epoch_record):
        with open(epoch_record, 'r') as f:
            count_episode = int(f.read())
    player_A = RunfastGameEnv()
    player_B = RunfastGameEnv()
    player_C = RunfastGameEnv()
    player_A.update_next(player_B)
    player_B.update_next(player_C)
    player_C.update_next(player_A)
    player_A.update_position('player_A')
    player_B.update_position('player_B')
    player_C.update_position('player_C')

    start_game = 'y'
    if start_game == 'y' or start_game == 'Y' or start_game == 'YES' or start_game == 'yes' or start_game == 'Yes':
        is_playing = True
    else:
        is_playing = False
    while(is_playing):
        a, b, c = divide_cards()
        player_A.cards_used  = np.zeros(13)
        player_B.cards_used = np.zeros(13)
        player_C.cards_used = np.zeros(13)

        ### save the result every 100,000 episodes
        if count_episode % ite_num == 0:
            torch.save(RL, model_file)
            print("##### Epoch: %s. #####" % count_episode)
            with open(epoch_record, 'w') as f:
                f.write(str(count_episode))
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
                action_cards = RL.choose_action(current_player)

                # implement Action
                small_cards = trans_vector(action_cards)
                current_player.cards_used = current_player.cards_used + small_cards
                biggestcards = action_cards
                if len(biggestcards)==4 and int(biggestcards[0]/4)==int(biggestcards[3]/4):
                    if int(biggestcards[0]/4)==int(biggestcards[3]/4) and (biggestcards[0]/4<11):
                        current_player.current_pattern = 3

                RL.store_transition(current_player, action_cards, current_state)
                current_state = current_player.get_state()
                isEnd = current_player.play_cards(action_cards)
                current_pattern = current_player.current_pattern
                ## Q-learning, calculate reward and add reward
                if isEnd:
                    # record that if the cards played lastest are boom
                    if len(biggestcards) == 4:
                        if int(biggestcards[0] / 4) == int(biggestcards[3] / 4) and (biggestcards[0] / 4 < 11):
                            current_player.boom_success = current_player.boom_success + 1
                    RL.add_reward(current_player, player_A, player_B, player_C)
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
                    action_cards = RL.choose_action(current_player, ways_toplay=ways_toplay)
                    small_cards = trans_vector(action_cards)
                    current_player.cards_used = current_player.cards_used + small_cards

                    biggestcards = action_cards
                    if len(biggestcards) == 4:
                        if int(biggestcards[0] / 4) == int(biggestcards[3] / 4) and (biggestcards[0] / 4 < 11):
                            current_pattern = 3
                    privious_player = current_player
                    RL.store_transition(current_player, action_cards, current_state)
                    current_state = current_player.get_state()
                    isEnd = current_player.play_cards(action_cards)
                    # current_pattern = current_player.current_pattern
                    ## Q-learning, calculate reward and add reward,winner reward = 1,loser = 0
                    if isEnd:
                        if len(biggestcards) == 4:
                            if int(biggestcards[0] / 4) == int(biggestcards[3] / 4) and (biggestcards[0] / 4 < 11):
                                current_player.boom_success = current_player.boom_success + 1
                        RL.add_reward(current_player, player_A, player_B, player_C)

                    if isEnd:
                        count_episode = count_episode+1
                        break

                    else:
                        current_player = current_player.get_next_player()


        if RL.memory_counter % RL.MEMORY_CAPACITY == 0 :
            RL.learn()



        # print('是否继续游戏，请输入y或者n：')
        if count_episode <= 1000000:
            start_game = 'y'
        if count_episode % 120000 == 0 and EPSILON <= 0.9:
            EPSILON = EPSILON + 0.01