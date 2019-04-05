import numpy as np
import random
import copy


class RunfastGameEnv():

    def __init__(self, cards=[], position=0, next_player=0, pattern=0):
        self.position = position
        self.next_player = next_player
        self.cards = cards
        self.pattern = pattern
        self.set_dict()
        self.cards_used = np.zeros(13)
        self.boom_success = 0
        self.status = np.array([0])
        self.current_pattern = 0

    def set_dict(self):
        number2card = {}
        card2number = {}
        j = 0
        for i in range (3,11):
            number2card.update({j:"♠"+str(i)})
            card2number.update({"♠" + str(i):j})
            j = j + 1
            number2card.update({j: "♥" + str(i)})
            card2number.update({"♠" + str(i): j})
            j = j + 1
            number2card.update({j: "♣" + str(i)})
            card2number.update({"♠" + str(i): j})
            j = j + 1
            number2card.update({j: "♦" + str(i)})
            card2number.update({"♠" + str(i): j})
            j = j + 1

        number2card.update({j: "♠J"})
        card2number.update({"♠J": j})
        j = j + 1
        number2card.update({j: "♥J"})
        card2number.update({"♥J": j})
        j = j + 1
        number2card.update({j: "♣J"})
        card2number.update({"♣J": j})
        j = j + 1
        number2card.update({j: "♦J"})
        card2number.update({"♦J": j})
        j = j + 1

        number2card.update({j: "♠Q"})
        card2number.update({"♠Q": j})
        j = j + 1
        number2card.update({j: "♥Q"})
        card2number.update({"♥Q": j})
        j = j + 1
        number2card.update({j: "♣Q"})
        card2number.update({"♣Q": j})
        j = j + 1
        number2card.update({j: "♦Q"})
        card2number.update({"♦Q": j})
        j = j + 1

        number2card.update({j: "♠K"})
        card2number.update({"♠K": j})
        j = j + 1
        number2card.update({j: "♥K"})
        card2number.update({"♥K": j})
        j = j + 1
        number2card.update({j: "♣K"})
        card2number.update({"♣K": j})
        j = j + 1
        number2card.update({j: "♦K"})
        card2number.update({"♦K": j})
        j = j + 1

        number2card.update({j: "♠A"})
        card2number.update({"♠A": j})
        j = j + 1
        number2card.update({j: "♥A"})
        card2number.update({"♥A": j})
        j = j + 1
        number2card.update({j: "♣A"})
        card2number.update({"♣A": j})
        j = j + 1
        number2card.update({j: "♦2"})
        card2number.update({"♦2": j})
        self.number2card = number2card
        self.card2number = card2number

    def cards2vec(self):
        card_vec = [0] * 13
        for i in self.cards:
            if i < 47:
                index = int(i / 4)
                card_vec[index] += 1
            else:
                card_vec[12] = 1
        return card_vec

    def set_cards(self, cards):
        self.cards = cards

    def get_cards(self):
        return self.cards

    def get_next_player(self):
        return self.next_player

    def get_position(self):
        return self.position

    def show_cards(self):
        print(self.position+"'s cards:")
        cards = []
        for i in self.cards:
            cards.append(self.number2card[i])
        print(cards)

    def update_position(self, new_position):
        self.position = new_position

    def update_next(self, new_next):
        self.next_player = new_next

    def check_long(self, lenth):
        cards_vec = self.cards2vec()
        current_len = 0
        for i in range(len(cards_vec) - lenth):
            current_len = 0
            for j in range(lenth):
                if cards_vec[i + j] >= 1:
                    current_len += 1
                    if current_len == lenth:
                        break
                else:
                    current_len = 0
                    break
            if current_len == lenth:
                return True
        return False

    # 检查连对子、飞机等情况
    def check_plane(self, lenth, width):
        cards_vec = self.cards2vec()
        current_len = 0
        for i in range(len(cards_vec) - lenth):
            current_len = 0
            for j in range(lenth):
                if cards_vec[i + j] >= width:
                    current_len += 1
                    if current_len == lenth:
                        break
                else:
                    current_len = 0
                    break
            if current_len == lenth:
                return True
        return False

    def search_pattern(self):
        pattern_to_playcards = []
        # pattern 0 代表出单牌
        pattern_to_playcards.append(0)
        # pattern 1 代表出对子


        cards_vec = self.cards2vec()

        for i in cards_vec:
            if i>=2:
                pattern_to_playcards.append(1)
                break

        #pattern 2或32 代表出三张
        for i in cards_vec:
            if i>=3:
                #如果不想让三带二先出，则注释下面这个添加32的操作
                #pattern_to_playcards.append(32)
                pattern_to_playcards.append(2)
                break

        #pattern 3 代表炸弹
        for i in cards_vec:
            if i==4:
                pattern_to_playcards.append(3)
                break

        # pattern 4 代表五张牌的顺子
        lenth = 5
        if self.check_long(lenth):
            pattern_to_playcards.append(4)

        # pattern 5代表6张牌的顺子
        lenth = 6
        if self.check_long(lenth):
            pattern_to_playcards.append(5)

        # pattern 6 代表7张牌的顺子
        lenth = 7
        if self.check_long(lenth):
            pattern_to_playcards.append(6)

        # pattern 7 代表8张牌的顺子
        lenth = 8
        if self.check_long(lenth):
            pattern_to_playcards.append(7)

        # pattern 8 代表9张牌的顺子
        lenth = 9
        if self.check_long(lenth):
            pattern_to_playcards.append(8)

        # pattern 9 代表10张牌的顺子
        lenth = 10
        if self.check_long(lenth):
            pattern_to_playcards.append(9)

        # pattern 10 代11张牌的顺子
        lenth = 11
        if self.check_long(lenth):
            pattern_to_playcards.append(10)

        # pattern 11 代表12张牌的顺子
        lenth = 12
        if self.check_long(lenth):
            pattern_to_playcards.append(11)

        #pattern 12 代表2连对
        lenth = 2
        width =2
        if self.check_plane(lenth, width):
            pattern_to_playcards.append(12)

        #pattern 13 代表3连对
        lenth = 3
        width =2
        if self.check_plane(lenth, width):
            pattern_to_playcards.append(13)

        #pattern 14 代表4连对
        lenth = 4
        width =2
        if self.check_plane(lenth, width):
            pattern_to_playcards.append(14)

        #pattern 15 代表5连对
        lenth = 5
        width =2
        if self.check_plane(lenth, width):
            pattern_to_playcards.append(15)

        #pattern 16 代表6连对
        lenth = 6
        width =2
        if self.check_plane(lenth, width):
            pattern_to_playcards.append(16)

        #pattern 17 代表7连对
        lenth = 7
        width =2
        if self.check_plane(lenth, width):
            pattern_to_playcards.append(17)

        #pattern 18 代表8连对
        lenth = 8
        width =2
        if self.check_plane(lenth, width):
            pattern_to_playcards.append(18)

        #pattern 19 代表二连飞机
        lenth = 2
        width =3
        if self.check_plane(lenth, width):
            pattern_to_playcards.append(19)

        #pattern 20 代表三连飞机
        lenth = 3
        width =3
        if self.check_plane(lenth, width):
            pattern_to_playcards.append(20)

        # pattern 21 代表四连飞机
        lenth = 4
        width = 3
        if self.check_plane(lenth, width):
            pattern_to_playcards.append(21)

        return pattern_to_playcards


    #得到连对和飞机等出法
    def get_air_solution(self,the_lenth=2,the_width=2, biggest=[-1,-1,-1,-1,-1]):
        lenth = the_lenth
        width = the_width
        current_len = 0
        cash_card = []
        cards_vec = self.cards2vec()
        way_to_playcards = []
        if biggest[0] == -1:
            for i in range(len(cards_vec) - lenth):
                current_len = 0
                for j in range(lenth):
                    if cards_vec[i + j] >= width:
                        current_len += 1
                        if current_len == lenth:
                            break
                    else:
                        current_len = 0
                        break
                if current_len == lenth:
                    base = i
                    count = 0
                    cash_card = []
                    for element in self.cards:
                        if int(element / 4) == base:
                            cash_card.append(element)
                            count += 1
                            if count==width:
                                base += 1
                                count = 0

                            if len(cash_card) == lenth*width:
                                way_to_playcards.append(cash_card)
                                break
        else:
            base_line = int(biggest[0] / 4) + 1
            if base_line >= len(cards_vec) - lenth:
                pass
            else:
                for i in range(base_line, len(cards_vec) - lenth):
                    current_len = 0
                    for j in range(lenth):
                        if cards_vec[i + j] >= width:
                            current_len += 1
                            if current_len == lenth:
                                break
                        else:
                            current_len = 0
                            break
                    if current_len == lenth:
                        base = i
                        cash_card = []
                        count = 0
                        for element in self.cards:
                            if int(element / 4) == base:
                                cash_card.append(element)
                                count += 1
                                if count==width:
                                    base += 1
                                    count = 0
                                if len(cash_card) == lenth*width:
                                    way_to_playcards.append(cash_card)
                                    break
        if width == 2:
            return way_to_playcards
        elif width == 3:
            if len(way_to_playcards)>0:
                if biggest[0] == -1:
                    if len(self.cards) <= 5*lenth:
                        choice_cards = list(set(self.cards) - set(way_to_playcards[0]))
                        if len(choice_cards)>0:
                            choice_cards.sort(reverse=False)
                            for element in choice_cards:
                                way_to_playcards[0].append(element)
                            return way_to_playcards[0]
                        else:
                            return way_to_playcards[0]
                    else:
                        return_result = []
                        for index in range(len(way_to_playcards)):
                            choice_cards = list(set(self.cards) - set(way_to_playcards[index]))

                            for i_i in range(10):
                                appendage = random.sample(choice_cards, lenth*2)
                                appendage.sort(reverse=False)
                                a = copy.deepcopy(way_to_playcards[index])
                                for element in appendage:
                                    a.append(element)
                                return_result.append(a)
                        return return_result
                else:
                    if len(self.cards) < 5*lenth:
                        return []
                    elif len(self.cards) == 5*lenth:
                        choice_cards = list(set(self.cards) - set(way_to_playcards[0]))
                        if len(choice_cards) > 0:
                            choice_cards.sort(reverse=False)
                            for element in choice_cards:
                                way_to_playcards[0].append(element)
                            return way_to_playcards[0]
                    else:
                        return_result = []
                        for index in range(len(way_to_playcards)):
                            choice_cards = list(set(self.cards) - set(way_to_playcards[index]))

                            for i_i in range(10):
                                appendage = random.sample(choice_cards, lenth * 2)
                                appendage.sort(reverse=False)
                                a = copy.deepcopy(way_to_playcards[index])
                                for element in appendage:
                                    a.append(element)
                                return_result.append(a)
                        return return_result
            else:
                return []


    #得到顺子的各种出法
    def get_solution(self,the_lenth=5,biggest=[-1,-1,-1,-1,-1]):
        lenth = the_lenth
        current_len = 0
        cash_card = []
        cards_vec = self.cards2vec()
        way_to_playcards = []
        if biggest[0] == -1:
            for i in range(len(cards_vec) - lenth):
                current_len = 0
                for j in range(lenth):
                    if cards_vec[i + j] >= 1:
                        current_len += 1
                        if current_len == lenth:
                            break
                    else:
                        current_len = 0
                        break
                if current_len == lenth:
                    base = i
                    cash_card = []
                    for element in self.cards:
                        if int(element / 4) == base:
                            cash_card.append(element)
                            base += 1
                            if len(cash_card) == lenth:
                                way_to_playcards.append(cash_card)
                                break
        else:
            base_line = int(biggest[0] / 4) + 1
            if base_line >= len(cards_vec) - lenth:
                pass
            else:
                for i in range(base_line, len(cards_vec) - lenth):
                    current_len = 0
                    for j in range(lenth):
                        if cards_vec[i + j] >= 1:
                            current_len += 1
                            if current_len == lenth:
                                break
                        else:
                            current_len = 0
                            break
                    if current_len == lenth:
                        base = i
                        cash_card = []
                        for element in self.cards:
                            if int(element / 4) == base:
                                cash_card.append(element)
                                base += 1
                                if len(cash_card) == lenth:
                                    way_to_playcards.append(cash_card)
                                    break
        return way_to_playcards


    def update_pattern(self, new_pattern):
        self.pattern = new_pattern

    def search_play_methods(self, pattern=0, biggest=[-1,-1,-1,-1,-1]):
        way_to_playcards = []
        #出单牌
        if pattern==0:
            if len(self.get_next_player().get_cards())==1:
                big = biggest[0]
                if big == -1:
                    base = self.cards[0]
                elif big > 43 and big < 47:
                    base = 47
                else:
                    base = (int(big / 4) + 1) * 4
                if self.cards[-1]>=base:
                    way_to_playcards.append([self.cards[-1] ])
            else:
                big = biggest[0]
                all_cards = self.cards
                if big==-1:
                    base = self.cards[0]
                elif big>43 and big<47:
                    base = 47
                else:
                    base = (int(big/4)+1)*4
                for i in all_cards:
                    list_a = []
                    if i>=base:
                        list_a.append(i)
                        way_to_playcards.append(list_a)
                        if base > 43 and base < 47:
                            base = 47
                        else:
                            base = (int(base / 4) + 1) * 4

        elif pattern==1:
            if biggest[0]==-1:
                base = 0
            else:
                base = int(biggest[0] / 4) + 1
            if len(self.cards)<=1:
                pass
            else:
                for i in range(len(self.cards)-1):
                    if int(self.cards[i]/4)>=base and int(self.cards[i]/4)==int(self.cards[i+1]/4) and self.cards[i+1]!=47:
                        way_to_playcards.append([self.cards[i], self.cards[i+1] ])
                        base = int(self.cards[i]/4)+1

        #出三张模式最高,之前是2，改为了32
        elif pattern==2:
        # elif pattern==32 or pattern==2:
            if biggest[0]==-1:
                base = 0
            else:
                base = int(biggest[0] / 4) + 1
            if len(self.cards)<=2:
                pass
            elif biggest[0]==-1 and len(self.cards)>=3:
                for i in range(len(self.cards)-2):
                    if int(self.cards[i]/4)>=base and int(self.cards[i]/4)==int(self.cards[i+2]/4) and self.cards[i+2]!=47:
                        base_list = [self.cards[i], self.cards[i + 1], self.cards[i + 2]]
                        #choice_cards = list(set(self.cards) - set(base_list))
                        if len(self.cards)==3:
                            way_to_playcards.append([self.cards[i], self.cards[i + 1], self.cards[i + 2]])
                        elif len(self.cards)==4:
                            choice_cards = list(set(self.cards) - set(base_list))
                            way_to_playcards.append([self.cards[i], self.cards[i + 1], self.cards[i + 2], choice_cards[0]])
                        elif len(self.cards)==5:
                            choice_cards = list(set(self.cards) - set(base_list))
                            if choice_cards[0] < choice_cards[1]:
                                way_to_playcards.append([self.cards[i], self.cards[i + 1], self.cards[i + 2], choice_cards[0], choice_cards[1]])
                            else:
                                way_to_playcards.append(
                                    [self.cards[i], self.cards[i + 1], self.cards[i + 2], choice_cards[1],
                                     choice_cards[0]])
                        else:
                            choice_cards = list(set(self.cards) - set(base_list))

                            for i_i in range(10):
                                random.shuffle(choice_cards)
                                if choice_cards[0] < choice_cards[1]:
                                    way_to_playcards.append(
                                        [self.cards[i], self.cards[i + 1], self.cards[i + 2], choice_cards[0],
                                         choice_cards[1]])
                                else:
                                    way_to_playcards.append(
                                        [self.cards[i], self.cards[i + 1], self.cards[i + 2], choice_cards[1],
                                         choice_cards[0]])
                            # way_to_playcards.append([self.cards[i], self.cards[i+1], self.cards[i+2] ])
                            base = int(self.cards[i] / 4) + 1
            elif biggest[0]!=-1 and len(self.cards)<5:
                pass

            elif biggest[0]!=-1 and len(self.cards)>=5:
                base = int(biggest[0] / 4) + 1
                for i in range(len(self.cards)-2):
                    if int(self.cards[i]/4)>=base and int(self.cards[i]/4)==int(self.cards[i+2]/4) and self.cards[i+2]!=47:
                        base_list = [self.cards[i], self.cards[i+1], self.cards[i+2]]
                        choice_cards = list(set(self.cards) - set(base_list))

                        for i_i in range(10):
                            random.shuffle(choice_cards)
                            if choice_cards[0] < choice_cards[1]:
                                way_to_playcards.append([self.cards[i], self.cards[i + 1], self.cards[i + 2], choice_cards[0], choice_cards[1]])
                            else:
                                way_to_playcards.append(
                                    [self.cards[i], self.cards[i + 1], self.cards[i + 2], choice_cards[1],
                                     choice_cards[0]])
                            if len(self.cards) == 5:
                                break
                        #way_to_playcards.append([self.cards[i], self.cards[i+1], self.cards[i+2] ])
                        base = int(self.cards[i]/4)+1



        # 是不是四张牌
        elif pattern==3:
            #是不是炸弹
            if biggest[0]==-1:
                Cards2vec = self.cards2vec()
                for i in range(len(Cards2vec)):
                    if Cards2vec[i] == 4:
                        a = [i * 4 + j for j in range(4)]
                        way_to_playcards.append(a)
            else:
                base = int(biggest[0]/4)
                Cards2vec = self.cards2vec()
                for i in range(len(Cards2vec)):
                    if i > base and i < 11:
                        if Cards2vec[i] == 4:
                            a = [i * 4 + j for j in range(4)]
                            way_to_playcards.append(a)
        # 是不是顺子
        elif pattern==4:
            lenth = 5
            result = self.get_solution(lenth,biggest)
            if len(result)>0:
                for element in result:
                    way_to_playcards.append(element)
        # 六张牌的顺子
        elif pattern==5:
            lenth = 6
            result = self.get_solution(lenth,biggest)
            if len(result)>0:
                for element in result:
                    way_to_playcards.append(element)
        #7张牌的顺子
        elif pattern==6:
            lenth = 7
            result = self.get_solution(lenth,biggest)
            if len(result)>0:
                for element in result:
                    way_to_playcards.append(element)
        #8张牌的顺子
        elif pattern==7:
            lenth = 8
            result = self.get_solution(lenth,biggest)
            if len(result)>0:
                for element in result:
                    way_to_playcards.append(element)
        #9张牌的顺子
        elif pattern==8:
            lenth = 9
            result = self.get_solution(lenth,biggest)
            if len(result)>0:
                for element in result:
                    way_to_playcards.append(element)
        #10张牌的顺子
        elif pattern==9:
            lenth = 10
            result = self.get_solution(lenth,biggest)
            if len(result)>0:
                for element in result:
                    way_to_playcards.append(element)
        #11张牌的顺子
        elif pattern==10:
            lenth = 11
            result = self.get_solution(lenth,biggest)
            if len(result)>0:
                for element in result:
                    way_to_playcards.append(element)
        #12张牌的顺子
        elif pattern==11:
            lenth = 12
            result = self.get_solution(lenth,biggest)
            if len(result)>0:
                for element in result:
                    way_to_playcards.append(element)

        #如果出的是连对
        elif pattern>11 and pattern<19:
            the_width = 2
            if biggest[0]==-1:
                the_lenth = pattern-10
            else:
                the_lenth = int(len(biggest)/the_width)
            result = self.get_air_solution(the_lenth, the_width, biggest)
            if len(result)>0:
                for element in result:
                    way_to_playcards.append(element)

        elif pattern >= 19:
            the_width = 3
            if biggest[0]==-1:
                the_lenth = pattern-17
            else:
                the_lenth = int(len(biggest)/the_width)
            result = self.get_air_solution(the_lenth, the_width, biggest)
            if len(result)>0:
                for element in result:
                    way_to_playcards.append(element)

        if pattern!=3 and biggest[0]!=-1:
            Cards2vec = self.cards2vec()
            for i in range(len(Cards2vec)):
                if Cards2vec[i] == 4:
                    a = [i * 4 + j for j in range(4)]
                    way_to_playcards.append(a)

        return way_to_playcards

    def play_cards(self, cards_toplay, notTry=True):
        if len(cards_toplay)==0:
            if notTry:
                print(self.position + '要不起！')
            return False
        cards_toplay_zu = ''
        for i in cards_toplay:
            cards_toplay_zu = cards_toplay_zu + self.number2card[i] + ','
        if notTry:
            print(self.position+' 出'+cards_toplay_zu)
        self.cards = list(set(self.cards) - set(cards_toplay))
        if len(self.cards)==0:
            if notTry:
                print(self.position+" wins.")
            return True
        self.cards.sort(reverse=False)
        return False