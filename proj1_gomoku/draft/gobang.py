import numpy as np
import random
import time
import re

COLOR_BLACK = 2
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)
max_deep = 2

five = ['11111', '22222']
live4 = ['(?<=0)1111(?=0)', '(?<=0)2222(?=0)']
live3 = ['(?<=0)0111(?=0)|(?<=0)1110(?=0)', '(?<=0)0222(?=0)|(?<=0)2220(?=0)']
live2 = ['(?<=0)11(?=0)', '(?<=0)22(?=0)']
form4 = ['(?<=[23])1111(?=0)|(?<=0)1111(?=[23])', '(?<=[13])2222(?=0)|(?<=0)2222(?=[13])']
sleep3 = ['(?<=[23])111(?=0)|(?<=0)111(?=[23])', '(?<=[13])222(?=0)|(?<=0)222(?=[13])']
sleep2 = ['(?<=[23])11(?=0)|(?<=0)11(?=[23])', '(?<=[13])22(?=0)|(?<=0)22(?=[13])']
dead4 = ['(?<=[23])1111(?=[23])|(?<=[23])1111(?=[23])', '(?<=[13])2222(?=[13])|(?<=[13])2222(?=[13])']
dead3 = ['(?<=[23])111(?=[23])|(?<=[23])111(?=[23])', '(?<=[13])222(?=[13])|(?<=[13])222(?=[13])']
dead2 = ['(?<=[23])11(?=[23])|(?<=[23])11(?=[23])', '(?<=[13])22(?=[13])|(?<=[13])22(?=[13])']
SCORE = np.array([0, 1000, 500, 280, 600, 300, 100, 50, 20, 10])

idx_5 = 0
idx_live4 = 1
idx_live3 = 2
idx_live2 = 3
idx_form4 = 4
idx_sleep3 = 5
idx_sleep2 = 6
idx_dead4 = 7
idx_dead3 = 8
idx_dead2 = 9


# don't change the class name
class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        # You are white or black
        if color == 1:
            self.color = color
            self.hum_color = 2
        else:
            self.color = 2
            self.hum_color = 1
        # the max time you should use, your algorithm's run time must not exceed the time
        self.time_out = time_out
        # You need add your decision into your candidate_list. System will get the end of your candidate_list as your decision .
        self.candidate_list = []
        self.win = 0  # win is 1 lose is -1
        self.defend_set = []
        self.attack_set = []

    def go(self, chessboard):
        # Clear candidate_list
        start = time.time()
        self.candidate_list.clear()
        chessboard = np.array(chessboard)
        chessboard[chessboard == -1] = 2
        temp = None
        # ==================================================================
        # Write your algorithm here
        # Here is the simplest sample:Random decision
        idx = np.where(chessboard == COLOR_NONE)
        idx = list(zip(idx[0], idx[1]))
        if len(idx) == self.chessboard_size ** 2 and self.color == COLOR_BLACK:
            new_pos = (self.chessboard_size >> 1, self.chessboard_size >> 1)
            self.candidate_list.append(new_pos)
        elif len(idx) == self.chessboard_size ** 2 - 1:
            new_pos = ((self.chessboard_size >> 1) - 1, (self.chessboard_size >> 1) - 1)
            self.candidate_list.append(new_pos)
        else:
            pos_idx = random.randint(0, len(idx) - 1)
            new_pos = idx[pos_idx]
            self.candidate_list.append(new_pos)
            next_color = 1
            if self.color == 2:
                next_color = -1
            temp = self.alpha_beta(chessboard, max_deep, idx, next_color, float('-inf'), float('inf'))
            new_pos = temp[1]
            # ==============Find new pos========================================
            # Make sure that the position of your decision in chess board is empty.
            # If not, return error.
            assert chessboard[new_pos[0], new_pos[1]] == COLOR_NONE
            # Add your decision into candidate_list, Records the chess board
            self.candidate_list.append(new_pos)
        chessboard[new_pos[0]][new_pos[1]] = self.color
        print(chessboard)
        print(time.time() - start)
        return temp

    def alpha_beta(self, chessboard, deep, idx, next_color, alpha, beta):
        if deep == 0:
            return [self.evaluate(chessboard), ()]
        new_pos = (0, 0)
        for i in range(0, len(idx)):
            v = idx[i]
            n, m = v[0], v[1]
            idx.pop(i)
            if next_color == 1:
                chessboard[n][m] = 1
            else:
                chessboard[n][m] = 2
            temp = self.alpha_beta(chessboard, deep - 1, idx, -next_color, -beta, -alpha)
            value = -temp[0]
            chessboard[n][m] = COLOR_NONE
            idx.insert(i, v)
            if value >= beta:
                return [beta, v]
            if value > alpha:
                alpha = value
                new_pos = v
        return [alpha, new_pos]

    # If our color is the color in args
    # then calculate total scores we may get in this chessboard
    def evaluate(self, chessboard) -> int:
        self.win = 0
        num = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        # calculate the rows
        self.calculate(chessboard, 0, num)
        # calculate the columns
        self.calculate(chessboard, 1, num)
        # calculate the diagonals
        self.calculate(chessboard, 2, num)
        self.calculate(chessboard, 3, num)
        com_scores = sum(num[0] * SCORE)
        hum_scores = sum(num[1] * SCORE)
        # print('num ' + str(num[0]) + ' score ' + str(com_scores))
        score = com_scores - hum_scores
        # print('com '+str(num_com)+' score '+str(com_scores))
        # print('hum '+str(num_hum)+' score '+str(hum_scores))
        if num[0][0] > 0 or num[0][1] > 0:
            self.win = 1
            score = float('inf')
        elif num[1][0] > 0 or num[1][1] > 0:
            self.win = -1
            score = float('-inf')
        return score

    def calculate(self, chessboard, direction, num):
        right = 0
        if direction in (0, 1):
            right = self.chessboard_size
        elif direction in (2, 3):
            right = 2 * self.chessboard_size - 9
            constant = self.chessboard_size - 5
        anti_chessboard = chessboard[:, ::-1]
        for i in range(0, right):
            if direction == 0:
                line = chessboard[i]
            elif direction == 1:
                line = chessboard[0:self.chessboard_size, i]
            elif direction == 2:
                line = chessboard.diagonal(i - constant)
            else:
                line = anti_chessboard.diagonal(i - constant)
            line = list(line)
            line.insert(0, 3)
            line.append(3)
            line_str = ''.join(str(count) for count in line)
            form = self.color - 1
            hum_form = self.hum_color - 1
            num[0] = num[0] + np.array(
                [len(re.findall(five[form], line_str)), len(re.findall(live4[form], line_str)),
                 len(re.findall(live3[form], line_str)), len(re.findall(live2[form], line_str)),
                 len(re.findall(form4[form], line_str)), len(re.findall(sleep3[form], line_str)),
                 len(re.findall(sleep2[form], line_str)), len(re.findall(dead4[form], line_str)),
                 len(re.findall(dead3[form], line_str)), len(re.findall(dead2[form], line_str))])
            num[1] = num[1] + np.array(
                [len(re.findall(five[hum_form], line_str)), len(re.findall(live4[hum_form], line_str)),
                 len(re.findall(live3[hum_form], line_str)), len(re.findall(live2[hum_form], line_str)),
                 len(re.findall(form4[hum_form], line_str)), len(re.findall(sleep3[hum_form], line_str)),
                 len(re.findall(sleep2[hum_form], line_str)), len(re.findall(dead4[hum_form], line_str)),
                 len(re.findall(dead3[hum_form], line_str)), len(re.findall(dead2[hum_form], line_str))])




cs15 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
cs10 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 2, 2, 2, 2, 0, 0, 0],
                 [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 ])
cs = cs10

ai = AI(len(cs), 2, 1)
print(cs)
print('============================')
print(ai.go(cs))
# print(ai.evaluate(cs))
