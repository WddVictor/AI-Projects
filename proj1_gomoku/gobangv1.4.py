import numpy as np
import random
import time

COLOR_BLACK = 2
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)
max_deep = 1

SCORE = np.array([32, 16, 8, 8, 3, 3, 2, 2, 1])

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
     '000010': 7, '000012': 8, '010000': 7, '100000': 8, '100001': 8, '100002': 8, '200001': 8, '210000': 8}
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
     '000020': 7, '000021': 8, '020000': 7, '100002': 8, '120000': 8, '200000': 8, '200001': 8, '200002': 8}

]


# don't change the class name
class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.chessboard = np.zeros((chessboard_size, chessboard_size)).astype('int')
        # You are white or black
        self.types = [np.zeros((2, chessboard_size, 9)).astype('int'),
                      np.zeros((2, chessboard_size, 9)).astype('int'),
                      np.zeros((2, 2 * chessboard_size - 1, 9)).astype('int'),
                      np.zeros((2, 2 * chessboard_size - 1, 9)).astype('int')]
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
        chessboard = np.array(chessboard).astype('int')
        chessboard[chessboard == -1] = 2
        print(chessboard)

        # ==================================================================
        # Write your algorithm here
        # Here is the simplest sample:Random decision
        pre_steps = np.where(self.chessboard - chessboard != 0)
        pre_steps = list(zip(pre_steps[0], pre_steps[1]))
        idx = np.where(chessboard == COLOR_NONE)
        idx = list(zip(idx[0], idx[1]))
        if len(pre_steps) != 0:
            types = self.types
            for step in pre_steps:
                types = self.calculate(chessboard[step],step,chessboard,types)
            self.types = types

        if len(idx) == self.chessboard_size ** 2 and self.color == COLOR_BLACK:
            n, m = self.chessboard_size >> 1, self.chessboard_size >> 1
            new_pos = (n, m)
            self.candidate_list.append(new_pos)
        elif len(idx) == self.chessboard_size ** 2 - 1:
            index = pre_steps[0]
            if index[0] in (self.chessboard_size - 1, 0) or index[1] in (self.chessboard_size, 0):
                new_pos = (self.chessboard_size >> 1, self.chessboard_size >> 1)
            else:
                new_pos = (index[0] - 1, index[1] - 1)
            self.candidate_list.append(new_pos)
        else:
            pos_idx = random.randint(0, len(idx) - 1)
            new_pos = idx[pos_idx]
            self.candidate_list.append(new_pos)
            temp = self.alpha_beta(chessboard, max_deep, idx, self.hum_color, float('-inf'), float('inf'), (),self.types.copy())
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
        self.types = self.calculate(self.color, new_pos, chessboard, self.types)
        self.chessboard = chessboard
        print(chessboard)
        print(new_pos)
        print(time.time() - start)

    def alpha_beta(self, chessboard, deep, idx, pre_color, alpha, beta, pre_step,pre_types):
        if deep == 0:
            a = self.evaluate(chessboard, pre_step, pre_color, self.types.copy())
            return [a, ()]
        new_pos = None
        if pre_color == self.hum_color:
            for i in range(0, len(idx)):
                v = idx[i]
                n, m = v[0], v[1]
                idx.pop(i)
                chessboard[n][m] = self.color
                types = pre_types.copy()
                types = self.calculate(self.color,v,chessboard,types)
                temp = self.alpha_beta(chessboard, deep - 1, idx, self.color, alpha, beta, v,types)
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
                types = pre_types.copy()
                types = self.calculate(self.hum_color, v, chessboard, types)
                temp = self.alpha_beta(chessboard, deep - 1, idx, self.hum_color, alpha, beta, v,types)
                chessboard[n][m] = COLOR_NONE
                idx.insert(i, v)
                if temp[0] < beta:
                    beta = temp[0]
                    new_pos = v
                if alpha >= beta:
                    break
            return [beta, new_pos]

    def evaluate(self, chessboard, pre_step, pre_color, pre_types) -> int:
        next_color = 1 if pre_color == 2 else 2
        types = pre_types.copy()
        types = self.calculate(pre_color, pre_step, chessboard, types)
        types = self.calculate(next_color, pre_step, chessboard, types)

        chessboard[pre_step[0]][pre_step[1]] = COLOR_NONE
        com_form = self.color - 1
        hum_form = self.hum_color - 1
        score = 0
        num = np.zeros((2, 9)).astype('int')
        for i in range(0, 4):
            num[com_form] += sum(types[i][com_form])
            num[hum_form] += sum(types[i][hum_form])
        if pre_step == (6,2):
            print(num)
        if num[com_form][idx_5] != 0:
            score += 10000
        elif num[hum_form][idx_live4] != 0 or num[hum_form][idx_form4] != 0:
            score -= 10000
        elif num[com_form][idx_live4] != 0 or num[com_form][idx_form4] > 1:
            score += 2000
        elif num[com_form][idx_form4] != 0:
            if num[com_form][idx_live3] != 0:
                if num[hum_form][idx_live3] == 0:
                    if num[hum_form][idx_sleep3] != 0:
                        score += 100
                    else:
                        score += 1000
                else:
                    score += 20
        elif num[hum_form][idx_live3] != 0:
            score -= 2000
        elif num[com_form][idx_live3] > 1:
            if num[hum_form][idx_sleep3] != 0:
                score += 50
            else:
                score += 500
        score += sum(num[com_form] * SCORE) - sum(num[hum_form] * SCORE)
        return score

    def calculate(self, as_color, pre_step, chessboard, types):
        form = as_color - 1
        n = pre_step[0]
        m = pre_step[1]
        anti_chessboard = chessboard[:, ::-1]
        for i in range(0, 4):
            if i == 0:
                line = chessboard[n]
                index = n
            elif i == 1:
                line = chessboard[0:len(chessboard), m]
                index = m
            elif i == 2:
                line = chessboard.diagonal(m - n)
                index = m - n + self.chessboard_size - 1
            else:
                new_m = len(chessboard[0]) - m - 1
                line = anti_chessboard.diagonal(new_m - n)
                index = new_m - n + self.chessboard_size - 1
            types[i][form, index] = types[i][form, index] * 0
            line_str = ''.join(str(count) for count in line)
            priority = 100
            last_type_index = -5
            if len(line_str) == 5:
                if forms[form].__contains__(line_str):
                    types[i][form, index, forms[form][line_str]] += 1
            for count in range(0, len(line_str) - 5):
                line_bries = line_str[count:count + 6]
                if forms[form].__contains__(line_bries):
                    temp = forms[form][line_bries]
                    if count >= last_type_index + 5 and priority != 100:
                        types[i][form, index, priority] += 1
                        priority = 100
                    if temp < priority:
                        priority = temp
                        last_type_index = count
            if priority != 100:

                types[i][form, index, priority] += 1
        return types


cs15 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0],
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
cs_ept = np.zeros((15, 15)).astype('int')
#
ai = AI(len(cs), 1, 1)
# print(cs)
# # print('============================')
print(ai.go(cs))
# print(ai.evaluate(cs, (1, 1), 2, ai.types))
