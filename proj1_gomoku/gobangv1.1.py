import numpy as np
import random
import time

COLOR_BLACK = 2
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)
max_deep = 1

SCORE = np.array([100000000, 100001, 5001, 601, 281, 301, 101, 51, 21])
SCORE_1 = np.array([10000000, 100000, 5000, 599, 280, 300, 100, 50, 20])

idx_5 = 0
idx_live4 = 1
idx_form4 = 2
idx_live3 = 3
idx_sleep3 = 4
idx_live2 = 5
idx_sleep2 = 6
idx_live1 = 7
idx_sleep1 = 8
forms = [
    {'11111': 0, '011111': 0, '111110': 0, '111111': 0, '111112': 0, '211111': 0, '011110': 1, '001111': 2, '010111': 2,
     '10111': 2, '11101': 2, '11011': 2,
     '011011': 2, '011101': 2, '011112': 2, '101110': 2, '101111': 2, '101112': 2, '110110': 2, '110111': 2,
     '110112': 2, '111010': 2, '111011': 2, '111012': 2, '111100': 2, '111101': 2, '111102': 2, '201111': 2,
     '210111': 2, '211011': 2, '211101': 2, '211110': 2, '001110': 3, '010110': 3, '011010': 3, '011100': 3,
     '000111': 4, '001011': 4, '001112': 4, '010011': 4, '010101': 4, '011001': 4, '011102': 4, '100110': 4,
     '100111': 4, '100112': 4, '101010': 4, '101011': 4, '101012': 4, '110010': 4, '110011': 4, '110012': 4,
     '110100': 4, '110101': 4, '111000': 4, '111001': 4, '111002': 4, '200111': 4, '201110': 4, '210011': 4,
     '210101': 4, '211001': 4, '211100': 4, '001010': 5, '001100': 5, '010010': 5, '010100': 5, '000011': 6,
     '000101': 6, '000110': 6, '000112': 6, '001001': 6, '010012': 6, '011000': 6, '100011': 6, '100100': 6,
     '100101': 6, '100102': 6, '101000': 6, '101001': 6, '110000': 6, '110001': 6, '110002': 6, '200011': 6,
     '201001': 6, '210010': 6, '211000': 6, '000100': 7, '001000': 7, '001002': 7, '200100': 7, '000001': 8,
     '000010': 8, '000012': 8, '010000': 8, '100000': 8, '100001': 8, '100002': 8, '200001': 8, '210000': 8}
    ,
    {'22222': 0, '022222': 0, '122222': 0, '222220': 0, '222221': 0, '222222': 0, '022220': 1, '002222': 2, '020222': 2,
     '20222': 2, '22202': 2, '22022': 2, '22200': 4,
     '022022': 2, '022202': 2, '022221': 2, '102222': 2, '120222': 2, '122022': 2, '122202': 2, '122220': 2,
     '202220': 2, '202221': 2, '202222': 2, '220220': 2, '220221': 2, '220222': 2, '222020': 2, '222021': 2,
     '222022': 2, '222200': 2, '222201': 2, '222202': 2, '002220': 3, '020220': 3, '022020': 3, '022200': 3,
     '000222': 4, '002022': 4, '002221': 4, '020022': 4, '020202': 4, '022002': 4, '022201': 4, '100222': 4,
     '102220': 4, '120022': 4, '120202': 4, '122002': 4, '122200': 4, '200220': 4, '200221': 4, '200222': 4,
     '202020': 4, '202021': 4, '202022': 4, '220020': 4, '220021': 4, '220022': 4, '220200': 4, '220202': 4,
     '222000': 4, '222001': 4, '222002': 4, '002020': 5, '002200': 5, '020020': 5, '020200': 5, '000022': 6,
     '000202': 6, '000220': 6, '000221': 6, '002002': 6, '020021': 6, '022000': 6, '100022': 6, '102002': 6,
     '120020': 6, '122000': 6, '200022': 6, '200200': 6, '200201': 6, '200202': 6, '202000': 6, '202002': 6,
     '220000': 6, '220001': 6, '220002': 6, '000200': 7, '002000': 7, '002001': 7, '100200': 7, '000002': 8,
     '000020': 8, '000021': 8, '020000': 8, '100002': 8, '120000': 8, '200000': 8, '200001': 8, '200002': 8}

]


# don't change the class name
class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        # You are white or black

        self.flag = 0

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
        self.pre_score = 0

    def go(self, chessboard):
        # Clear candidate_list
        start = time.time()
        self.candidate_list.clear()
        chessboard = np.array(chessboard)
        chessboard[chessboard == -1] = 2
        # ==================================================================
        # Write your algorithm here
        # Here is the simplest sample:Random decision
        idx = np.where(chessboard == COLOR_NONE)
        idx = list(zip(idx[0], idx[1]))
        if len(idx) == self.chessboard_size ** 2 and self.color == COLOR_BLACK:
            new_pos = (self.chessboard_size >> 1, self.chessboard_size >> 1)
            self.candidate_list.append(new_pos)
        elif len(idx) == self.chessboard_size ** 2 - 1:
            pre_step = np.where(chessboard != 0)
            index = (pre_step[0][0], pre_step[1][0])
            if index[0] in (self.chessboard_size - 1, 0) or index[1] in (self.chessboard_size, 0):
                new_pos = (self.chessboard_size >> 1, self.chessboard_size >> 1)
            else:
                new_pos = (index[0] - 1, index[1] - 1)
            self.candidate_list.append(new_pos)
        else:
            pos_idx = random.randint(0, len(idx) - 1)
            new_pos = idx[pos_idx]
            self.candidate_list.append(new_pos)
            temp = self.alpha_beta(chessboard, max_deep, idx, self.hum_color, float('-inf'), float('inf'), ())
            new_pos = temp[1]
            # ==============Find new pos========================================
            # Make sure that the position of your decision in chess board is empty.
            # If not, return error.
            try:
                assert chessboard[new_pos[0], new_pos[1]] == COLOR_NONE
            except Exception:
                print(new_pos, " fuc")
            # Add your decision into candidate_list, Records the chess board
            self.candidate_list.append(new_pos)
        chessboard[new_pos[0]][new_pos[1]] = self.color
        print(time.time() - start)

    def alpha_beta(self, chessboard, deep, idx, pre_color, alpha, beta, pre_step):
        if deep == 0:
            a = self.evaluate(chessboard, pre_step, pre_color)
            return [a, ()]
        new_pos = None
        if pre_color == self.hum_color:
            for i in range(0, len(idx)):
                v = idx[i]
                n, m = v[0], v[1]
                idx.pop(i)
                chessboard[n][m] = self.color
                temp = self.alpha_beta(chessboard, deep - 1, idx, self.color, alpha, beta, v)
                if temp[0] == float('inf'):
                    self.flag = 1
                chessboard[n][m] = COLOR_NONE
                idx.insert(i, v)
                if temp[0] > alpha:
                    alpha = temp[0]
                    new_pos = v
                if alpha >= beta:
                    break
            return [alpha, new_pos]
        elif pre_color == self.color:
            for i in range(0, len(idx)):
                v = idx[i]
                n, m = v[0], v[1]
                idx.pop(i)
                chessboard[n][m] = self.hum_color
                temp = self.alpha_beta(chessboard, deep - 1, idx, self.hum_color, alpha, beta, v)
                chessboard[n][m] = COLOR_NONE
                idx.insert(i, v)
                if temp[0] < beta:
                    beta = temp[0]
                    new_pos = v
                if alpha >= beta:
                    break
            return [beta, new_pos]

    def evaluate(self, chessboard, pre_step, pre_color) -> int:
        next_color = 1 if pre_color == 2 else 2
        num = np.zeros((2, 9)).astype('int')
        self.calculate(pre_color, pre_step, chessboard, num)
        chessboard[pre_step[0]][pre_step[1]] = next_color
        self.calculate(next_color, pre_step, chessboard, num)
        chessboard[pre_step[0]][pre_step[1]] = COLOR_NONE
        pre_form = pre_color - 1
        other_form = 0
        if pre_form == 0:
            other_form = 1
        pre_score = sum(num[pre_form] * SCORE_1)
        other_score = sum(num[other_form] * SCORE)
        score = pre_score + other_score
        if num[pre_form][idx_5] != 0:
            score += float('inf')
        elif num[other_form][idx_5] != 0:
            score += 1000000000
        elif num[pre_form][idx_live4] != 0:
            score += 10000000
        elif num[other_form][idx_live4] != 0:
            score += 5000000
        elif num[pre_form][idx_form4] > 1:
            score += 4000000
        elif num[other_form][idx_form4] > 1:
            score += 3000000
        elif num[pre_form][idx_live3] != 0 and num[pre_form][idx_form4] != 0:
            score += 2900000
        elif num[other_form][idx_live3] != 0 and num[other_form][idx_form4] != 0:
            score += 2800000
        elif num[pre_form][idx_live3] > 1:
            score += 2000000
        elif num[other_form][idx_live3] > 1:
            score += 1500000
        if pre_color == self.color:
            return score
        else:
            return -score

    def calculate(self, as_color, pre_step, chessboard, num):
        form = as_color - 1
        n = pre_step[0]
        m = pre_step[1]
        smaller_board = chessboard[max(0, n - 4):min(self.chessboard_size, n + 5),
                        max(0, m - 4):min(self.chessboard_size, m + 5)]
        if n - 4 > 0:
            n = 4
        if m - 4 > 0:
            m = 4
        anti_chessboard = smaller_board[:, ::-1]
        for i in range(0, 4):
            if i == 0:
                line = smaller_board[n]
            elif i == 1:
                line = smaller_board[0:len(smaller_board), m]
            elif i == 2:
                line = smaller_board.diagonal(m - n)
            else:
                new_m = len(smaller_board[0]) - m - 1
                line = anti_chessboard.diagonal(new_m - n)
            line_str = ''.join(str(count) for count in line)
            priority = 100
            if len(line_str) == 5:
                if forms[form].__contains__(line_str):
                    num[form][forms[form][line_str]] += 1
            for count in range(0, len(line_str) - 5):
                line_bries = line_str[count:count + 6]
                if forms[form].__contains__(line_bries):
                    temp = forms[form][line_bries]
                    if temp < priority:
                        priority = temp
            if priority != 100:
                num[form][priority] += 1


cs15 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 2, 2, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

cs10 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
                 [0, 1, 1, 1, 1, 2, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 ])
cs = cs15
# ai = AI(len(cs), 2, 1)
# print(cs)
# print('============================')
# print(ai.go(cs))
# print(ai.evaluate(cs,(5,6),1))
