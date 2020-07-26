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

        char_data = self.char_data
        n, row, col = char_data.shape     

        # メッシュ特徴量格納用(2000, 8, 8)
        mesh_feature = np.zeros((n, int(row/step), int(col/step)))

        # 特徴量の計算
        for i in range(row//step):
            for j in range(col//step):
                mesh_feature[:,i,j] = np.sum(
                    char_data[:, i*step:(i+1)*step, j*step:(j+1)*step], axis=(1,2)) / float(step**2)

        return mesh_feature.reshape(n, -1) # 2000x64

    def _read_data(self):
        """ 手書き文字のデータを読み込む"""

        # データ格納用
        char_data = []
        for i in range(len(Data.chars)):

            # 読み込むファイルの名前
            f_name = "hira{}_{:02d}{}.dat".format(self.writer, i, "L" if self.is_train else "T")

            with open(os.path.join(self.data_dir, f_name)) as f:

                # 2次元のint配列に変換(shape: 6400x64)
                c = np.array([[int(c) for c in l] 
                    for l in f.read().strip().split('\n')], dtype="uint8")

                h, w = c.shape                              # 6400x64
                char_data.append(c.reshape(int(h/w), w, w)) # 10x64x64

        return np.concatenate(char_data, 0)  # 2000x64x64

    def __str__(self):
        return "writer_{} {}".format(self.writer, "Train" if self.is_train else "Test")

