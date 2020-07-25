#!/usr/bin/python3
import cv2
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from Data import Data
from neural_network import MiddleLayer, OutputLayer

def show_char(char, label):
    "char: numpy.array(0,1)"
    cv2.imshow(label, char * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_chars(chars, labels):
    "char: numpy.array(0,1)"
    for c, l in zip(chars, labels):
        show_char(c, l)

# クロスエントロピー誤差
def cross_entropy(t, batch_size):
    return -np.sum(t * np.log(output_layer.y + 1e-7)) / batch_size

# 正答率
def accuracy(input_, correct):
    middle_layer.forward_prop(input_)
    output_layer.forward_prop(middle_layer.y)

    cnt_correct = np.sum(np.argmax(output_layer.y, axis=1) == np.argmax(correct, axis=1))
    return cnt_correct / input_.shape[0] * 100
    

if __name__ == "__main__":
    # 文字の種類
    chars = ['a',  'i',  'u',  'e',  'o',
             'ka', 'ki', 'ku', 'ke', 'ko',
             'sa', 'si', 'su', 'se', 'so',
             'ta', 'ti', 'tu', 'te', 'to',]

    # データの読み込み
    data_train_0 = Data(data_dir="Data", writer=0, is_train=True)
    data_test_0 = Data(data_dir="Data", writer=0, is_train=False)
    data_train_1 = Data(data_dir="Data", writer=1, is_train=True)
    data_test_1 = Data(data_dir="Data", writer=1, is_train=False)

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


    # 設定値
    n_input = 64    # 入力層のニューロン数
    n_middle = 64   # 中間層のニューロン数
    n_output = 20   # 出力層のニューロン数

    eta = 0.01      # 学習係数
    alpha = 0.5     # 安定化係数
    epoch = 151    # エポック数
    batch_size = 8  # バッチサイズ(ミニバッチ学習を行う)
    interval = 5  # 学習状態の表示間隔

    # 各層の初期化
    middle_layer = MiddleLayer(n_input, n_middle, eta, alpha)
    output_layer = OutputLayer(n_middle, n_output, eta, alpha)

    
    # 誤差の記録
    train_error_x = []
    train_error_y = []
    test_error_x = []
    test_error_y = []

    n_batch = n_train // batch_size # 1epochあたりのバッチ数

    for i in range(epoch):
        
        # 誤差の計算
        middle_layer.forward_prop(input_train)
        output_layer.forward_prop(middle_layer.y)
        error_train = cross_entropy(correct_train, n_train)

        middle_layer.forward_prop(input_test)
        output_layer.forward_prop(middle_layer.y)
        error_test = cross_entropy(correct_test, n_test)

        # 誤差の記録
        train_error_x.append(i)
        train_error_y.append(error_train)
        test_error_x.append(i)
        test_error_y.append(error_test)


        # 学習経過
        if i % interval == 0:
            print("Epoch: {}\nError_train: {}\nError_test: {}".format(i, error_train, error_test))
            print("Accuracy train: {}\nAccuracy test: {}".format(
                accuracy(input_train, correct_train),
                accuracy(input_test, correct_test)))
            print("=" * 50)

        # 学習
        index_random = np.arange(n_train)
        np.random.shuffle(index_random)
        for j in range(n_batch):
            
            # ミニバッチ作成
            mb_index = index_random[j*batch_size: (j+1)*batch_size]
            x = input_train[mb_index, :]   # 入力
            t = correct_train[mb_index, :] # 正解
    
            # 順伝播
            middle_layer.forward_prop(x)
            output_layer.forward_prop(middle_layer.y)

            # 逆伝播
            output_layer.back_prop(t)
            middle_layer.back_prop(output_layer.grad_x)

            # 重みとバイアスの更新
            middle_layer.update()
            output_layer.update()


    # グラフの表示
    plt.plot(train_error_x, train_error_y, label="train")
    plt.plot(test_error_x, test_error_y, label="test")
    plt.legend()

    plt.xlabel("Epoch")
    plt.ylabel("Error")

    plt.show()


    # 画像で確認
    middle_layer.forward_prop(input_test)
    output_layer.forward_prop(middle_layer.y)

    # chars_correct = # 正解
    #  = [chars[idx] for idx in np.argmax(output_layer.y, axis=1)] # 予測値

    show_chars(data_test_0.char_data, labels)
