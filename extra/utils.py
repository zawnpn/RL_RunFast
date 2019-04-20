import random
import torch
import numpy as np

from extra.config import empty_vec

# if torch.cuda.is_available():
#     torch.cuda.set_device(0)


def avg_score(a, b, c):
    m = np.mean([a, b, c])
    s_max = np.max([a, b, c])
    s_min = np.min([a, b, c])
    a = (a - m) / (s_max - s_min)
    b = (b - m) / (s_max - s_min)
    c = (c - m) / (s_max - s_min)
    return a, b, c

# 发牌
def divide_cards():
    cards = [i for i in range(48)]
    random.shuffle(cards)
    a_cards = cards[:16]
    b_cards = cards[16:32]
    c_cards = cards[32:]
    a_cards.sort()
    b_cards.sort()
    c_cards.sort()
    return a_cards, b_cards, c_cards


# 将48维0-1向量变为13维向量
def trans_vector(cards):
    if type(cards) == int:
        cards = [cards]
    small_cards = np.zeros(13)
    for i in range(len(cards)):
        temp = (cards[i])//4
        small_cards[temp] = small_cards[temp]+1
    return small_cards


# 函数trans_cards_to_new_representation和get_cards_small_extend用来将，表示本次出牌的13维向量即动作a进行扩展
def trans_cards_to_new_representation(cards, cards_belong_dan, cards_belong_dui, cards_belong_3_or_4, pattern, control):
    if pattern == 0:
        new_representation =  np.concatenate((cards, empty_vec, empty_vec, empty_vec, empty_vec, empty_vec,
                                              empty_vec, empty_vec, empty_vec, empty_vec, empty_vec), axis=0)
    elif pattern == 1:
        new_representation = np.concatenate((empty_vec, cards, empty_vec, empty_vec, empty_vec, empty_vec,
                                             empty_vec, empty_vec, empty_vec, empty_vec, empty_vec), axis=0)
    # positions means (cards_belong_dan), (cards_belong_dui), (cards_belong_3 or 4)
    elif pattern == 2:
        new_representation = np.concatenate((empty_vec, empty_vec, cards, empty_vec, empty_vec, empty_vec, empty_vec,
                                             (cards_belong_dan), (cards_belong_dui), empty_vec, empty_vec), axis=0)
    elif pattern == 3:
        new_representation = np.concatenate((empty_vec, empty_vec, empty_vec, cards, empty_vec, empty_vec, empty_vec,
                                             empty_vec, empty_vec, empty_vec, empty_vec), axis=0)
    elif 4 <= pattern <= 11:
        new_representation = np.concatenate((empty_vec, empty_vec, empty_vec, empty_vec, cards, empty_vec, empty_vec,
                                             empty_vec, empty_vec, empty_vec, empty_vec), axis=0)
    elif 11 < pattern < 19:
        new_representation = np.concatenate((empty_vec, empty_vec, empty_vec, empty_vec, empty_vec, cards, empty_vec,
                                             empty_vec, empty_vec, empty_vec, empty_vec), axis=0)
    # positions means (cards_belong_dan), (cards_belong_dui), (cards_belong_3 or 4)
    elif pattern == 19 and control != 5:
        new_representation = np.concatenate((empty_vec, empty_vec, empty_vec, empty_vec, empty_vec, empty_vec, cards,
                                             (cards_belong_dan), (cards_belong_dui), (cards_belong_3_or_4),
                                             empty_vec), axis=0)
    # positions means (cards_belong_dan), (cards_belong_dui), (cards_belong_3 or 4)
    elif pattern == 19 and control == 5:
        new_representation = np.concatenate((empty_vec, empty_vec, empty_vec, empty_vec, empty_vec, empty_vec, cards,
                                             empty_vec, empty_vec, empty_vec, (cards_belong_3_or_4)), axis=0)
    # positions means (cards_belong_dan), (cards_belong_dui), (cards_belong_3 or 4)
    elif pattern == 20:
        new_representation = np.concatenate((empty_vec, empty_vec, empty_vec, empty_vec, empty_vec, empty_vec, cards,
                                             empty_vec, empty_vec, empty_vec, (cards_belong_3_or_4)), axis=0)
    elif pattern > 20:
        new_representation = np.concatenate((empty_vec, empty_vec, empty_vec, empty_vec, empty_vec, empty_vec, cards,
                                             empty_vec, empty_vec, empty_vec, (cards_belong_3_or_4)), axis=0)
    return new_representation


def get_cards_small_extend(cards, pattern):
    if type(cards)==int:
        cards = [cards]
    control_for_cards_belong = -1
    if pattern == 2: # 三张带两张
        a = cards[:3]
        b = cards[3:]
        cards_small_a = trans_vector(a)
        if len(b)==2 :
            if b[0]/4 == b[1]/4 and b[1] != 47:
                # 带的牌是一对
                control_for_cards_belong = 1
                cards_small_b = trans_vector(b)
                cards_small_extended = trans_cards_to_new_representation(cards_small_a, empty_vec, cards_small_b,
                                                                         empty_vec, pattern, control_for_cards_belong)
            else:
                # 带的牌是两张单牌
                control_for_cards_belong = 0
                cards_small_b = trans_vector(b)
                cards_small_extended = trans_cards_to_new_representation(cards_small_a, cards_small_b, empty_vec,
                                                                         empty_vec, pattern, control_for_cards_belong)
        else:
            # 带一张单牌或者不带牌（手中的总牌数少于五张，可以直接将牌打空）
            control_for_cards_belong = 0
            cards_small_b = trans_vector(b)
            cards_small_extended = trans_cards_to_new_representation(cards_small_a, cards_small_b, empty_vec,
                                                                     empty_vec, pattern, control_for_cards_belong)
    elif pattern == 19:  # 二连飞机，六张带四张
        a = cards[:6]
        b = cards[6:]
        cards_small_a = trans_vector(a)
        cards_small_b = trans_vector(b)
        if len(b) == 4:
            if b[0]/4 != b[1]/4 and b[1]/4 != b[2]/4 and b[2]/4 != b[3]/4:
                # control==0, 带的牌是四张单牌
                control_for_cards_belong = 0
                cards_small_b = trans_vector(b)
                cards_small_extended = trans_cards_to_new_representation(cards_small_a, cards_small_b, empty_vec,
                                                                         empty_vec, pattern, control_for_cards_belong)
            elif b[0]/4 != b[1]/4 and b[1]/4 != b[2]/4 and b[3] == 47:
                control_for_cards_belong = 0
                cards_small_b = trans_vector(b)
                cards_small_extended = trans_cards_to_new_representation(cards_small_a, cards_small_b, empty_vec,
                                                                         empty_vec, pattern, control_for_cards_belong)
            elif b[0]/4 == b[1]/4 and b[1]/4 != b[2]/4 and b[2]/4 == b[3]/4 and b[3] != 47:
                # control==1, 带的牌是两对
                control_for_cards_belong = 2
                cards_small_b = trans_vector(b)
                cards_small_extended = trans_cards_to_new_representation(cards_small_a, empty_vec, cards_small_b,
                                                                         empty_vec, pattern, control_for_cards_belong)
            elif b[0]/4 == b[1]/4 and b[1]/4 == b[2]/4 and b[2]/4 != b[3]/4:
                # control==3, 带的牌是一个三张，一个单牌
                control_for_cards_belong = 3
                b_1 = b[:3]
                b_2 = b[3:4]
                # b_1 represent 1个三张, b_2 represent 1个单牌
                cards_small_b_1 = trans_vector(b_1)
                cards_small_b_2 = trans_vector(b_2)
                cards_small_extended = trans_cards_to_new_representation(cards_small_a, empty_vec,
                                                                         cards_small_b_2, cards_small_b_1, pattern,
                                                                         control_for_cards_belong)
            elif b[0]/4 == b[1]/4 and b[1]/4 == b[2]/4 and b[3] == 47:
                control_for_cards_belong = 3
                b_1 = b[:3]
                b_2 = b[3:4]
                # b_1 represent 1个三张, b_2 represent 1个单牌
                cards_small_b_1 = trans_vector(b_1)
                cards_small_b_2 = trans_vector(b_2)
                cards_small_extended = trans_cards_to_new_representation(cards_small_a, cards_small_b_2, empty_vec,
                                                                         cards_small_b_1, pattern,
                                                                         control_for_cards_belong)
            elif b[0]/4 != b[1]/4 and b[1]/4 == b[2]/4 and b[2]/4 == b[3]/4 and b[3] != 47:
                control_for_cards_belong = 3
                b_1 = b[:1]
                b_2 = b[1:4]
                # b_1 represent 1个单牌, b_2 represent 1个三张
                cards_small_b_1 = trans_vector(b_1)
                cards_small_b_2 = trans_vector(b_2)
                cards_small_extended = trans_cards_to_new_representation(cards_small_a, cards_small_b_1, empty_vec,
                                                                         cards_small_b_2, pattern,
                                                                         control_for_cards_belong)
            elif b[0]/4 == b[1]/4 and b[1]/4 == b[2]/4 and b[2]/4 == b[3]/4 and b[3] != 47:
                # control==4, 1个炸弹
                control_for_cards_belong = 4
                cards_small_b = trans_vector(b)
                cards_small_extended = trans_cards_to_new_representation(cards_small_a, empty_vec, empty_vec,
                                                                         cards_small_b, pattern,
                                                                         control_for_cards_belong)
            elif b[0]/4 == b[1]/4 and b[1]/4 == b[2]/4:
                # control==1, 1对和 2个单牌,[1,2]位置是对, [3,4]位置是两个单牌
                control_for_cards_belong = 1
                b_1 = b[:2]
                b_2 = b[2:4]
                # b_1 represent 1对, b_2 represent 1个三张
                cards_small_b_1 = trans_vector(b_1)
                cards_small_b_2 = trans_vector(b_2)
                cards_small_extended = trans_cards_to_new_representation(cards_small_a, cards_small_b_2,
                                                                         cards_small_b_1, empty_vec, pattern,
                                                                         control_for_cards_belong)
            elif b[1]/4 == b[1]/4 and b[2]/4 == b[2]/4:
                # control==1, 1对和2个单牌,[2,3]位置是对, [1,4]位置是两个单牌
                control_for_cards_belong = 1
                b_1 = b[1:3]
                b_2 = [b[0],b[3]]
                # b_1 represent 1对, b_2 represent 2个单牌
                cards_small_b_1 = trans_vector(b_1)
                cards_small_b_2 = trans_vector(b_2)
                cards_small_extended = trans_cards_to_new_representation(cards_small_a, cards_small_b_2,
                                                                         cards_small_b_1, empty_vec, pattern,
                                                                         control_for_cards_belong)
            else:
                # control==1, 1对和2个单牌
                control_for_cards_belong = 1
                b_1 = b[2:4]
                b_2 = b[:2]
                # b_1 represent 1 对, b_2 represent 2个单牌
                cards_small_b_1 = trans_vector(b_1)
                cards_small_b_2 = trans_vector(b_2)
                cards_small_extended = trans_cards_to_new_representation(cards_small_a, cards_small_b_2,
                                                                         cards_small_b_1, empty_vec, pattern,
                                                                         control_for_cards_belong)
        else:
            #  带牌不够四张（手中总牌数小于10，带牌不够四张，随意出牌即可胜利）
            control_for_cards_belong = 5
            cards_small_b = trans_vector(b)
            cards_small_extended = trans_cards_to_new_representation(cards_small_a, empty_vec, empty_vec,
                                                                     cards_small_b, pattern, control_for_cards_belong)
    elif pattern == 20:# 三连飞机，九张带六张，随意出牌即可胜利
        a = cards[:9]
        b = cards[9:]
        cards_small_a = trans_vector(a)
        cards_small_b = trans_vector(b)
        cards_small_extended = trans_cards_to_new_representation(cards_small_a, empty_vec, empty_vec, cards_small_b,
                                                                 pattern, control_for_cards_belong)
    else:# 四连飞机，12张带4张（共16张牌），随意出牌即可胜利
        cards_small = trans_vector(cards)
        cards_small_extended = trans_cards_to_new_representation(cards_small, empty_vec, empty_vec, empty_vec,
                                                                 pattern, control_for_cards_belong)

    return cards_small_extended


# 牌局介绍，根据剩牌计算输分作为reward
def calculate_score(cards_left):
    cards_number = cards_left.sum()
    # if cards_number == 1:
    #     score = np.array([0])
    # elif cards_number == 16:
    #     score = (-2) * cards_number
    # else:
    #     score = -cards_number
    # score = 0 - cards_number**2
    if cards_number > 0:
        score = -1
    else:
        score = 0
    return score
