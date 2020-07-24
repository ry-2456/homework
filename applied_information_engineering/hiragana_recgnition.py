#!/usr/bin/python3
import os
import sys
import time
import numpy as np

from Data import Data

def print_chars(char_data):
    """文字データの表示"""
    for i in range(len(char_data)):
        chars = char_data[i]
        n, row, col = chars.shape # 10x64x64
        for j in range(n):
            c = chars[j]
            for k in range(row):
                for l in range(col):
                    print(c[k, l], end="")
                print()

            print("\n" + "#"*64 + "\n")
            input()

if __name__ == "__main__":
    chars = ['a',  'i',  'u',  'e',  'o',
             'ka', 'ki', 'ku', 'ke', 'ko',
             'sa', 'si', 'su', 'se', 'so',
             'ta', 'ti', 'tu', 'te', 'to',]

    start = time.time()

    # データの読み込み
    data_train_0 = Data(data_dir="Data", writer=0, is_train=True)
    data_test_0 = Data(data_dir="Data", writer=0, is_train=False)
    data_train_1 = Data(data_dir="Data", writer=1, is_train=True)
    data_test_1 = Data(data_dir="Data", writer=1, is_train=False)

    print(data_train_0)
    print(data_test_0)
    print(data_train_1)
    print(data_test_1)
    end = time.time()
    print(end - start)
