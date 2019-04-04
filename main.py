import sys
import copy
import numpy as np
from extra.utils import divide_cards, trans_vector
from GameEnv.RunFastGame import RunfastGameEnv
from RL_framework.DQN import DQN
from extra.config import ite_num, saved_file, EPSILON, Select_LongestPattern_probability, count_episode

if __name__ == '__main__':
    RL = DQN()
    player_A = RunfastGameEnv()
    player_B = RunfastGameEnv()
    player_C = RunfastGameEnv()
    player_A.update_next(player_B)
    player_B.update_next(player_C)
    player_C.update_next(player_A)
    player_A.update_position('player_A')
    player_B.update_position('player_B')
    player_C.update_position('player_C')


    # print("开始游戏，请输入y或者n：")
    # start_game = input()
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

        flag_for_test = False
        ### save the result every 100,000 episodes
        if count_episode % ite_num == 0:
            flag_for_test  = True
            savedStdout = sys.stdout # save output straem
            card_flag = np.random.randint(3)
            if card_flag == 0:
                a = [6,11,12,13,14,15,18,21,23,27,28,35,39,41,43,44]
                b = [0,1,2,3,16,17,19,22,24,26,29,32,37,40,45,46]
                c = [4,5,7,8,9,10,20,25,30,31,33,34,36,38,42,47]
            elif card_flag == 1:
                a = [2,3,6,11,13,15,18,21,23,27,28,33,35,39,43,44]
                b = [0,4,8,12,16,17,19,22,26,29,32,37,40,41,45,46]
                c = [1,5,7,9,10,14,20,24,25,30,31,34,36,38,42,47]
            elif card_flag == 2:
                a = [4,6,10,11,18,21,23,27,28,35,36,38,39,41,43,44]
                b = [2,3,13,14,15,16,17,19,26,29,30,31,32,40,45,46]
                c = [0,1,5,7,8,9,12,20,22,24,25,33,34,37,42,47]
            file = open(saved_file, 'a+')
            sys.stdout = file
            print("this is test epoch: " + str(count_episode) )
            print("Select_LongestPattern_probability: ",Select_LongestPattern_probability)
            print("EPSILON: ", EPSILON)
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
            print("player_A plays first")
        elif 0 in b:
            current_player = player_B
            privious_player = player_B
            print("player_B plays first")
        elif 0 in c:
            current_player = player_C
            privious_player = player_C
            print("player_C plays first")
        isEnd = False

        current_pattern = 0
        biggestcards = []
        while(isEnd==False ):

            if current_player == privious_player:
                # record who play boom（炸弹） and boom_success+1
                if len(biggestcards) == 4:
                    if int(biggestcards[0] / 4) == int(biggestcards[3] / 4) and (biggestcards[0] / 4 < 11):
                        current_player.boom_success = current_player.boom_success + 1
                current_player.show_cards()
                ##### 停牌阶段展示一下各家手牌
                temp = current_player.get_next_player()
                temp.show_cards()
                temp = temp.get_next_player()
                temp.show_cards()
                biggestcards = []
                pattern_to_playcards = current_player.search_pattern()
                ting = np.array([1])
                cards, current_pattern, _ = RL.choose_action(current_player, ting, pattern_to_playcards)
                small_cards = trans_vector(cards)
                current_player.cards_used = current_player.cards_used + small_cards
                biggestcards = cards
                if len(biggestcards)==4 and int(biggestcards[0]/4)==int(biggestcards[3]/4):
                    if int(biggestcards[0]/4)==int(biggestcards[3]/4) and (biggestcards[0]/4<11):
                        current_pattern = 3

                #打出手牌
                ## Q-learning, store situation and cards in hand
                next_player = current_player.get_next_player()
                next_next_player = next_player.get_next_player()
                ting = np.array([1])
                RL.store_transition(cards, current_player.cards, current_player.cards_used, next_player.cards_used,
                                 next_next_player.cards_used, current_player.position, ting, current_pattern)
                ## Q-learning, store situation and cards in hand
                isEnd = current_player.play_cards(cards)
                ## Q-learning, calculate reward and add reward
                if isEnd:
                    # record that if the cards played lastest are boom
                    if len(biggestcards) == 4:
                        if int(biggestcards[0] / 4) == int(biggestcards[3] / 4) and (biggestcards[0] / 4 < 11):
                            current_player.boom_success = current_player.boom_success + 1
                    RL.add_reward(current_player.position, next_player.position, next_next_player.position, next_player.cards, next_next_player.cards, player_A, player_B, player_C)
                ## Q-learning, calculate reward and add reward
                if isEnd:
                    print("player_A remains " + str(len(player_A.cards)) + " cards.")
                    print("player_B remains " + str(len(player_B.cards)) + " cards.")
                    print("player_C remains " + str(len(player_C.cards)) + " cards.")
                    #print("happy ending 2")

                    if flag_for_test :
                        sys.stdout = savedStdout
                        print('')
                        print('')
                        print('')
                        print('')
                        file.close()

                    count_episode = count_episode+1
                    break

                else:
                    current_player = current_player.get_next_player()

            else:
                ways_toplay = current_player.search_play_methods(current_pattern, biggestcards)
                if len(ways_toplay)==0:
                    print(current_player.get_position() + '要不起！')
                    current_player = current_player.get_next_player()
                else:
                    ting = np.array([0])
                    cards, _, _ = RL.choose_action(current_player, ting, [current_pattern], follow=True, ways_toplay=ways_toplay)
                    small_cards = trans_vector(cards)
                    current_player.cards_used = current_player.cards_used + small_cards

                    biggestcards = cards
                    if len(biggestcards) == 4:
                        if int(biggestcards[0] / 4) == int(biggestcards[3] / 4) and (biggestcards[0] / 4 < 11):
                            current_pattern = 3
                    privious_player = current_player

                    ## Q-learning, store situation and cards in hand
                    next_player = current_player.get_next_player()
                    next_next_player = next_player.get_next_player()
                    ting = np.array([0])
                    RL.store_transition(cards, current_player.cards, current_player.cards_used, next_player.cards_used,
                                         next_next_player.cards_used, current_player.position, ting, current_pattern)
                    ## Q-learning, store situation and cards in hand
                    isEnd = current_player.play_cards(cards)
                    ## Q-learning, calculate reward and add reward,winner reward = 1,loser = 0
                    if isEnd:
                        if len(biggestcards) == 4:
                            if int(biggestcards[0] / 4) == int(biggestcards[3] / 4) and (biggestcards[0] / 4 < 11):
                                current_player.boom_success = current_player.boom_success + 1
                        RL.add_reward(current_player.position, next_player.position, next_next_player.position, next_player.cards,
                                       next_next_player.cards, player_A, player_B, player_C)
                    ## Q-learning, calculate reward and add reward

                    if isEnd:
                        print("player_A remains " + str(len(player_A.cards)) + " cards.")
                        print("player_B remains " + str(len(player_B.cards)) + " cards.")
                        print("player_C remains " + str(len(player_C.cards)) + " cards.")
                        #print("happy ending 1")
                        if flag_for_test:

                            sys.stdout = savedStdout
                            print('')
                            print('')
                            print('')
                            print('')
                            file.close()
                        count_episode = count_episode+1
                        break

                    else:
                        current_player = current_player.get_next_player()


        if RL.memory_counter % RL.MEMORY_CAPACITY == 0 :
            RL.learn()


        print('是否继续游戏，请输入y或者n：')
        if count_episode <= 1000000:
            start_game = 'y'
        if count_episode % 120000 == 0 and EPSILON <= 0.9:
            EPSILON = EPSILON + 0.01
        if count_episode % 350000 == 0 and Select_LongestPattern_probability >= 0.1:
            Select_LongestPattern_probability = Select_LongestPattern_probability - 0.1
        print(count_episode)