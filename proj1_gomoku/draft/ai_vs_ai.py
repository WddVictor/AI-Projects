#!/usr/bin/env python3
"""
check the security and functionability of uploaded code
- forbid from importing os
- random chessboard check
- some special case check
"""
import imp
import numpy as np

FORBIDDEN_LIST = ['import os', 'exec']


class CodeCheck():
    def __init__(self, script_file_path, chessboard_size):
        self.time_out = 1
        self.script_file_path = script_file_path
        self.chessboard_size = chessboard_size
        self.agent = None
        self.errormsg = 'Error'

def new_vs_old(new_ai, old_ai):
    cs = np.zeros((15, 15)).astype('int')
    while True:
        if new_ai.color == 2:
            new_ai.go(cs)
            new_pos = new_ai.candidate_list[-1]
            cs[new_pos] = new_ai.color
            print("new: ", new_pos)
            print(cs)
            if is_win(new_ai, new_pos, cs):
                print('============================')
                print("new !")
                break
            old_ai.go(cs)
            new_pos = old_ai.candidate_list[-1]
            cs[new_pos] = old_ai.color
            print("old: ", new_pos)
            print(cs)
            if is_win(old_ai, new_pos, cs):
                print('============================')
                print("old !")
                break
        elif old_ai.color == 2:
            old_ai.go(cs)
            new_pos = old_ai.candidate_list[-1]
            cs[new_pos] = old_ai.color
            print("old: ", new_pos)
            print(cs)
            if is_win(old_ai, new_pos, cs):
                print('============================')
                print("old !")
                break
            new_ai.go(cs)
            new_pos = new_ai.candidate_list[-1]
            cs[new_pos] = new_ai.color
            print("new: ", new_pos)
            print(cs)
            if is_win(new_ai, new_pos, cs):
                print('============================')
                print("new !")
                break


def is_win(ai, pre_step, chessboard):
    num = np.zeros((2,9)).astype('int')
    ai.calculate(ai.color, pre_step, chessboard, num)
    form = ai.color - 1
    return num[form][0] == 1


def main():
    old = CodeCheck("./gobangv1.1.py", 15)
    new = CodeCheck("./gobangv1.6.py", 15)

    new_color = -1



    new_ai = imp.load_source('AI', new.script_file_path).AI(new.chessboard_size, new_color,
                                                            new.time_out)
    old_ai = imp.load_source('AI', old.script_file_path).AI(old.chessboard_size, -new_color,
                                                            old.time_out)
    new_vs_old(new_ai, old_ai)

    # code_checker = new
    # if not code_checker.check_code():
    #     print(code_checker.errormsg)
    # else:
    #     print('pass')


if __name__ == '__main__':
    main()
