POISSON = 0.3       # ポアソン比
T = 1e-3            # 厚さ
E = 203e9           # ヤング率

F_X = -9e5             # x方向荷重
F_Y = -1e4          # y方向荷重
LOAD_POINT_X = 1    # 荷重点のx座標
FIXED_X = -1        # 固定端のx座標

DATA_FILE_NAME = "FEM_Data.dat" # データのファイル名

load_k = False # K matrixを読み込むかどうか
save_k = False # K matrixを保存するかどうか

load_u = False # 変位uを読み込むかどうか
save_u = False # 変位uを保存するかどうか
