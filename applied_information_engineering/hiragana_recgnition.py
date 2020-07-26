#!/usr/bin/python3
import cv2
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from config import *

from Data import Data
from neural_network import MiddleLayer, OutputLayer, NeuralNetwork

# 文字を画像表示
def show_char(char, label):
    "char: numpy.array(0,1)"
    cv2.imshow(label, char * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 複数の文字を画像表示
def show_chars(chars, labels):
    "char: numpy.array(0,1)"
    for c, l in zip(chars, labels):
        show_char(c, l)

def show_graph(train_error_x, train_error_y, test_error_x, test_error_y,  train_label, test_labels):
    # グラフの表示
    plt.plot(train_error_x, train_error_y, label=train_label) # 学習データのplot
    for i in range(len(test_error_y)):                    # テストデータのplot
        plt.plot(test_error_x, test_error_y[i], label=test_labels[i])

    plt.legend()

    plt.xlabel("Epoch")
    plt.ylabel("Error")

    plt.show()

def main_2(input_train, input_test, correct_train_orig, correct_test_orig, train_data_name, test_data_name):
    # input_train, correct_train: numpy.array
    # input_test, correct_test: list 要素はnumpy.arary

    n_train = len(input_train)            # 学習データの数
    n_test = len(input_test[0])              # テストデータの数

    # 正解データをone-hot表現に
    correct_train = np.zeros((n_train, len(chars)))                                 # 2000x20
    correct_test = [np.zeros((n_test, len(chars))) for i in range(len(input_test))] # 2000x20がlen(input_test)個

    for i in range(n_train):
        correct_train[i, correct_train_orig[i]] = 1

    for i in range(len(input_test)):
        for j in range(n_test):
            correct_test[i][j, correct_test_orig[i][j]] = 1

    # 各層の初期化
    middle_layer = MiddleLayer(n_input, n_middle, eta, alpha)
    output_layer = OutputLayer(n_middle, n_output, eta, alpha)

    # ネットワークの初期化
    net = NeuralNetwork([middle_layer, output_layer])
    
    # 誤差の記録
    train_error_x = []
    train_error_y = []
    test_error_x = []
    test_error_y = [[] for i in range(len(input_test))]

    n_batch = n_train // batch_size # 1epochあたりのバッチ数

    for i in range(epoch):
        
        # 誤差の計算
        net.forward_prop(input_train)                           # 順伝播
        error_train = net.cross_entropy(correct_train, n_train) # クロスエントロピー誤差の計算

        error_test = []
        for j in range(len(input_test)):
            net.forward_prop(input_test[j])                                # 順伝播
            error_test.append(net.cross_entropy(correct_test[j], n_test))  # クロスエントロピー誤差の計算

        # 誤差の記録
        train_error_x.append(i)
        train_error_y.append(error_train)
        test_error_x.append(i)
        for j in range(len(input_test)):
            test_error_y[j].append(error_test[j])

        # 学習経過
        if i % interval == 0:
            print("Epoch: {}".format(i))

            # クロスエントロピー誤差表示
            print("Error_train({}): {}".format(train_data_name, error_train))
            for j in range(len(input_test)):
                print("Error_test({}): {}".format(test_data_name[j], error_test[j]))
            
            # 正答率表示
            print("Accuracy_train({}): {}".format(
                train_data_name, net.accuracy(input_train, correct_train)))
            for j in range(len(input_test)):
                print("Accuracy_test({}): {}".format(
                    test_data_name[j], net.accuracy(input_test[j], correct_test[j])))
            print("=" * 50)

        # 学習
        index_random = np.arange(n_train)
        np.random.shuffle(index_random)
        for j in range(n_batch):
            
            # ミニバッチ作成
            mb_index = index_random[j*batch_size: (j+1)*batch_size]
            input_ = input_train[mb_index, :]    # 入力
            correct = correct_train[mb_index, :] # 正解

            net.forward_prop(input_)   # 順伝播
            net.back_prop(correct)     # 逆伝播
            net.update_wb()            # 重みとバイアスの更新

    # グラフの表示
    show_graph(train_error_x, train_error_y, 
               test_error_x, test_error_y,  
               train_data_name, test_data_name)


    # 画像で確認
    net.forward_prop(input_test)

    # chars_correct = # 正解
    #  = [chars[idx] for idx in np.argmax(output_layer.y, axis=1)] # 予測値

    show_chars(data_test_0.char_data, labels)

def main():
    input_train = data_train_0.input_data # 学習データ
    input_test = data_test_0.input_data   # テストデータ

    n_train = len(input_train)            # 学習データの数
    n_test = len(input_test)              # テストデータの数

    # 正解データをone-hot表現に
    correct_train = np.zeros((n_train, len(chars))) # 2000x20
    correct_test = np.zeros((n_test, len(chars))) # 2000x20

    for i in range(n_train):
        correct_train[i, data_train_0.correct_data[i]] = 1

    for i in range(n_test):
        correct_test[i, data_test_0.correct_data[i]] = 1

    # 各層の初期化
    middle_layer = MiddleLayer(n_input, n_middle, eta, alpha)
    output_layer = OutputLayer(n_middle, n_output, eta, alpha)

    # ネットワークの初期化
    net = NeuralNetwork([middle_layer, output_layer])
    
    # 誤差の記録
    train_error_x = []
    train_error_y = []
    test_error_x = []
    test_error_y = []

    n_batch = n_train // batch_size # 1epochあたりのバッチ数

    for i in range(epoch):
        
        # 誤差の計算
        net.forward_prop(input_train)                           # 順伝播
        error_train = net.cross_entropy(correct_train, n_train) # クロスエントロピー誤差の計算

        net.forward_prop(input_test)                            # 順伝播
        error_test = net.cross_entropy(correct_test, n_test)    # クロスエントロピー誤差の計算

        # 誤差の記録
        train_error_x.append(i)
        train_error_y.append(error_train)
        test_error_x.append(i)
        test_error_y.append(error_test)

        # 学習経過
        if i % interval == 0:
            print("interval")
            print("Epoch: {}\nError_train: {}\nError_test: {}".format(i, error_train, error_test))
            print("Accuracy train: {}\nAccuracy test: {}".format(
                net.accuracy(input_train, correct_train),
                net.accuracy(input_test, correct_test)))
            print("=" * 50)

        # 学習
        index_random = np.arange(n_train)
        np.random.shuffle(index_random)
        for j in range(n_batch):
            
            # ミニバッチ作成
            mb_index = index_random[j*batch_size: (j+1)*batch_size]
            input_ = input_train[mb_index, :]    # 入力
            correct = correct_train[mb_index, :] # 正解

            net.forward_prop(input_)   # 順伝播
            net.back_prop(correct)     # 逆伝播
            net.update_wb()            # 重みとバイアスの更新

    # グラフの表示
    plt.plot(train_error_x, train_error_y, label="train")
    plt.plot(test_error_x, test_error_y, label="test")
    plt.legend()

    plt.xlabel("Epoch")
    plt.ylabel("Error")

    plt.show()

    # 画像で確認
    net.forward_prop(input_test)

    # chars_correct = # 正解
    #  = [chars[idx] for idx in np.argmax(output_layer.y, axis=1)] # 予測値

    show_chars(data_test_0.char_data, labels)

if __name__ == "__main__":
    # 文字の種類
    chars = ['a',  'i',  'u',  'e',  'o',
             'ka', 'ki', 'ku', 'ke', 'ko',
             'sa', 'si', 'su', 'se', 'so',
             'ta', 'ti', 'tu', 'te', 'to',]

    while True:
        print("\n学習データの選択")
        mode = input("0: 筆記者0\n1: 筆記者1\n2: 筆記者0と筆記者1  > ")
        if mode in ['0', '1', '2']:
            mode = int(mode)
            break

    # データの読み込み
    data_train_0 = Data(data_dir="Data", writer=0, is_train=True)
    data_test_0 = Data(data_dir="Data", writer=0, is_train=False)
    data_train_1 = Data(data_dir="Data", writer=1, is_train=True)
    data_test_1 = Data(data_dir="Data", writer=1, is_train=False)

    # main()

    if mode == 0:
        # train: 筆記者0の学習データ  test:筆記者0の学習データ, 筆記者0のテストデータ, 筆記者1のテストデータ
        input_train = data_train_0.input_data
        input_test = [data_train_0.input_data, data_test_0.input_data, data_test_1.input_data]

        correct_train = data_train_0.correct_data
        correct_test = [data_train_0.correct_data, data_test_0.correct_data, data_test_1.correct_data]

        train_data_name = "writer_0 train"
        test_data_name = ["writer_0 train", "writer_0 test", "writer_1 test"]
        
    elif mode == 1:
        # train: 筆記者1の学習データ  test:筆記者1の学習データ, 筆記者0のテストデータ, 筆記者1のテストデータ
        input_train = data_train_1.input_data
        input_test = [data_train_1.input_data, data_test_0.input_data, data_test_1.input_data]

        correct_train = data_train_1.correct_data
        correct_test = [data_train_1.correct_data, data_test_0.correct_data, data_test_1.correct_data]

        train_data_name = "writer_1 train"
        test_data_name = ["writer_1 train", "writer_0 test", "writer_1 test"]

    elif mode == 2:
        # train: 筆記者0と筆記者1の学習データ test:筆記者0と筆記者1のテストデータ
        input_train = np.concatenate([data_train_0.input_data, data_train_1.input_data])
        input_test = [np.concatenate([data_test_0.input_data, data_test_1.input_data])]

        correct_train = np.concatenate([data_train_0.correct_data, data_train_1.correct_data])
        correct_test = [np.concatenate([data_test_0.correct_data, data_test_1.correct_data])]

        train_data_name = "writer_0 writer_1 train"
        test_data_name = ["writer_0 writer_1 test"]

    main_2(input_train, input_test, correct_train, correct_test, train_data_name, test_data_name) 
