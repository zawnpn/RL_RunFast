import sys
import numpy as np
import torch
from GameEnv.RunFastGame import RunfastGameEnv
from extra.config import test_file, model_A_file, model_B_file, model_C_file, test_num
from extra.utils import divide_cards, trans_vector

if __name__ == '__main__':
    RLA = torch.load(model_A_file)
    RLB = torch.load(model_B_file)
    RLC = torch.load(model_C_file)
    player_A = RunfastGameEnv()
    player_B = RunfastGameEnv()
    player_C = RunfastGameEnv()
    player_A.update_next(player_B)
    player_B.update_next(player_C)
    player_C.update_next(player_A)
    player_A.update_position('player_A')
    player_B.update_position('player_B')
    player_C.update_position('player_C')

    savedStdout = sys.stdout # save output straem
    file = open(test_file, 'w')
    sys.stdout = file
    awc = 0  # a_wins_count
    bwc = 0
    cwc = 0
    for test_count in range(test_num):
        a, b, c = divide_cards()
        player_A.cards_used  = np.zeros(13)
        player_B.cards_used = np.zeros(13)
        player_C.cards_used = np.zeros(13)
        print("Test epoch: " + str(test_count+1) )
        player_A.set_cards(a)
        player_B.set_cards(b)
        player_C.set_cards(c)
        # 选择初始玩家，黑桃三出头
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
        else:
            current_player = player_C
            privious_player = player_C
            print("player_C plays first")
        isEnd = False

        current_pattern = 0
        biggestcards = []
        while(isEnd==False ):

            if current_player == privious_player:
                current_player.status = np.array([1])
                # record who play boom（炸弹） and boom_success+1
                if len(biggestcards) == 4:
                    if int(biggestcards[0] / 4) == int(biggestcards[3] / 4) and (biggestcards[0] / 4 < 11):
                        current_player.boom_success = current_player.boom_success + 1
                current_player.show_cards()

                # 停牌阶段展示一下各家手牌
                temp = current_player.get_next_player()
                temp.show_cards()
                temp = temp.get_next_player()
                temp.show_cards()
                biggestcards = []

                # choose Action from State.
                if current_player.position == 'player_A':
                    action_cards = RLA.choose_action(current_player,0.99)
                elif current_player.position == 'player_B':
                    action_cards = RLB.choose_action(current_player,0.99)
                elif current_player.position == 'player_C':
                    action_cards = RLC.choose_action(current_player,0.99)

                # implement Action
                small_cards = trans_vector(action_cards)
                current_player.cards_used = current_player.cards_used + small_cards
                biggestcards = action_cards
                if len(biggestcards)==4 and int(biggestcards[0]/4)==int(biggestcards[3]/4):
                    if int(biggestcards[0]/4)==int(biggestcards[3]/4) and (biggestcards[0]/4<11):
                        current_player.current_pattern = 3

                isEnd = current_player.play_cards(action_cards, test=True)
                current_pattern = current_player.current_pattern
                ## Q-learning, calculate reward and add reward
                if isEnd:
                    # record that if the cards played lastest are boom
                    if len(biggestcards) == 4:
                        if int(biggestcards[0] / 4) == int(biggestcards[3] / 4) and (biggestcards[0] / 4 < 11):
                            current_player.boom_success = current_player.boom_success + 1
                ## Q-learning, calculate reward and add reward
                if isEnd:
                    if len(player_A.cards) == 0:
                        awc += 1
                    elif len(player_B.cards) == 0:
                        bwc += 1
                    elif len(player_C.cards) ==0:
                        cwc += 1
                    print("player_A remains " + str(len(player_A.cards)) + " cards.")
                    print("player_B remains " + str(len(player_B.cards)) + " cards.")
                    print("player_C remains " + str(len(player_C.cards)) + " cards.")
                    break

                else:
                    current_player = current_player.get_next_player()

            else:
                current_player.current_pattern = current_pattern
                ways_toplay = current_player.search_play_methods(current_pattern, biggestcards)
                if len(ways_toplay)==0:
                    print(current_player.get_position() + '要不起！')
                    current_player = current_player.get_next_player()
                else:
                    current_player.status = np.array([0])
                    # choose Action from State.
                    if current_player.position == 'player_A':
                        action_cards = RLA.choose_action(current_player, 0.99, ways_toplay=ways_toplay)
                    elif current_player.position == 'player_B':
                        action_cards = RLB.choose_action(current_player, 0.99, ways_toplay=ways_toplay)
                    elif current_player.position == 'player_C':
                        action_cards = RLC.choose_action(current_player, 0.99, ways_toplay=ways_toplay)
                    small_cards = trans_vector(action_cards)
                    current_player.cards_used = current_player.cards_used + small_cards

                    biggestcards = action_cards
                    if len(biggestcards) == 4:
                        if int(biggestcards[0] / 4) == int(biggestcards[3] / 4) and (biggestcards[0] / 4 < 11):
                            current_pattern = 3
                    privious_player = current_player
                    ## Q-learning, store situation and cards in hand
                    isEnd = current_player.play_cards(action_cards, test=True)
                    # current_pattern = current_player.current_pattern
                    ## Q-learning, calculate reward and add reward,winner reward = 1,loser = 0
                    if isEnd:
                        if len(biggestcards) == 4:
                            if int(biggestcards[0] / 4) == int(biggestcards[3] / 4) and (biggestcards[0] / 4 < 11):
                                current_player.boom_success = current_player.boom_success + 1
                    ## Q-learning, calculate reward and add reward

                    if isEnd:
                        if len(player_A.cards) == 0:
                            awc += 1
                        elif len(player_B.cards) == 0:
                            bwc += 1
                        elif len(player_C.cards) == 0:
                            cwc += 1
                        print("player_A remains " + str(len(player_A.cards)) + " cards.")
                        print("player_B remains " + str(len(player_B.cards)) + " cards.")
                        print("player_C remains " + str(len(player_C.cards)) + " cards.")
                        break

                    else:
                        current_player = current_player.get_next_player()
    sys.stdout = savedStdout
    file.close()
    print('\n%s test finished.' % test_num)
    print("A's win rate: %s/%s=%s" % (awc, test_num, awc / test_num))
    print("B's win rate: %s/%s=%s" % (bwc, test_num, bwc / test_num))
    print("C's win rate: %s/%s=%s" % (cwc, test_num, cwc / test_num))
