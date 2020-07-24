import numpy as np
import os

class Data:
    """データの読み込み，特徴量抽出を行う"""
    # 20種類の文字(あ~と: 0~19)

    chars = ['a',  'i',  'u',  'e',  'o',
             'ka', 'ki', 'ku', 'ke', 'ko',
             'sa', 'si', 'su', 'se', 'so',
             'ta', 'ti', 'tu', 'te', 'to',]

    def __init__(self, data_dir, writer, is_train):
        self.data_dir = data_dir               # 文字列データのディレクトリ
        self.writer = writer                   # 書き手
        self.is_train = is_train               # Train=>True  Test=>False
        self.char_data = self._read_data()     # 読み込んだひらがなデータ
        self.input_data = self._mesh_featuer() # メッシュ特徴量を計算
        self.correct_data = np.zeros(len(self.input_data), dtype="int8") # 正解データの初期化

        # 正解データの計算
        for i in range(len(Data.chars)):
            n = int(len(self.input_data) / len(Data.chars))
            self.correct_data[i*n:(i+1)*n] = i

    def _mesh_featuer(self, step=8):
        """データから特徴量を抽出する
        step : 特徴量抽出領域の大きさ
        """
        char_data = self._read_data() # key:0~19 val:100x64x64(numpy.array)
        mesh_featuer_list = []

        # 特徴量の計算
        for i in range(len(Data.chars)):

            chars = char_data[i] 
            n, row, col = chars.shape # 100x64x64

            for j in range(n):
                c = chars[j]
                mesh_f = np.zeros((step, step))
                for k in range(step):
                    for l in range(step):
                        mesh_f[k, l] = np.sum(c[k*step:(k+1)*step, l*step:(l+1)*step]) / float(step ** 2)
                mesh_featuer_list.append(mesh_f)

        return mesh_featuer_list

    def _read_data(self):
        """ 手書き文字のデータを読み込む"""
        char_data = {}
        for i in range(len(Data.chars)):

            f_name = "hira{}_{:02d}{}.dat".format(self.writer, i, "L" if self.is_train else "T")

            # with open(self.data_dir + os.sep + f_name) as f:
            with open(os.path.join(self.data_dir, f_name)) as f:
                # data[Data.chars[i]] = np.array([[int(c) for c in l] for l in f.read().strip().split('\n')], dtype="uint8")
                c = np.array([[int(c) for c in l] for l in f.read().strip().split('\n')], dtype="uint8") # shape:6400 x 64
                h, w = c.shape
                char_data[i] = c.reshape(int(h/w), w, w)

        return char_data  # key:0~19 val:np.array: 100x64x64 

    def __str__(self):
        return "Data_type: writer_{} {}".format(self.writer, "Train" if self.is_train else "Test")

