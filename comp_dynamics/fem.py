import os
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    import cv2
    show_img = True
except ImportError as e:
    show_img = False

load_k = False # K matrixを読むこむかどうか
save_k = False # K matrixを保存するかどうか

load_u = False # 変位uを読み込むかどうか
save_u = False # 変位uを保存するかどうか

def read_fem_data(file_name):
    """データを読み込む""" 
    with open(file_name) as f:
        
        node_n = int(f.readline().split()[0]) # 節点数
        elem_n = int(f.readline().split()[0]) # 要素数

        nodes = np.zeros((node_n+1, 2), dtype="float128") # 一行目はダミー
        elems = np.zeros((elem_n+1, 3), dtype="int16")    # 一行目はダミー

        # ノード物理量の読み込み
        for i in range(node_n):
            x, y = map(float, f.readline().split()[1:]) # 精度に問題あり
            nodes[i+1, :] = x, y

        # 要素の読み込み
        for i in range(elem_n):
            u1, u2, u3 = map(int, f.readline().split()[2:])
            elems[i+1, :] = u1, u2, u3

    return (node_n, elem_n, nodes, elems)    

# 処理の始まりにmsgを終わりにdoneを表示するdecorator
def progress_msg(msg):
    def decorator(f):
        def wrapper(*args):
            print(msg, end="", flush=True)
            result =  f(*args)
            print("done")
            return result
        return wrapper
    return decorator

def get_elem_area(nodes, elem):
    """
    elemの面積を返す
    -------------------
    nodes : 全節点
    elem  : 面積を求めたい要素
    """
    u1, u2, u3 = elem
    x1, y1 = nodes[u1] 
    x2, y2 = nodes[u2]
    x3, y3 = nodes[u3]

    s = np.abs(0.5 * (y1*(x2-x3) + y2*(x3-x1) + y3*(x1-x2)))

    return s

def B_matrix(nodes, elem):
    """
    elemのB_matrixを返す
    -------------------
    nodes : 全節点
    elem  : B_matrixを求めたい要素
    """
    u1, u2, u3 = elem
    x1, y1 = nodes[u1] 
    x2, y2 = nodes[u2]
    x3, y3 = nodes[u3]

    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2

    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1
    
    B_mat = np.array([[b1, 0, b2, 0, b3, 0],
                      [0, c1, 0, c2, 0, c3],
                      [c1, b1, c2, b2, c3, b3]])

    s = get_elem_area(nodes, elem)

    return B_mat / (2*s)

def D_matrix(E, v):
    """
    D_matrixを返す
    -----------------
    E : ヤング率
    v : ポアソン比
    """
    c = E / (1 - v ** 2) 
    D_mat = c * np.array([[1, v, 0],
                          [v, 1, 0],
                          [0, 0, (1-v)/2.0]])
    return D_mat

def k_elem_matrix(B, D, S, t):
    """
    要素剛性マトリクスを返す
    -------------------
    B : B_matrix
    D : D_matrix
    S : 要素の面積
    t : 要素の厚さ
    """
    k = S * t * B.T.dot(D).dot(B)
    return k

@progress_msg("calculating K matrix...")
def K_matrix(k, n_node, n_elem, nodes, elems):
    """
    全体剛性マトリクスを返す
    ------------------------
    k      : すべての要素剛性マトリクス
    n_node : 節点数
    n_elem : 要素数
    nodes  : 節点
    elems  : 要素
    """
    K = np.zeros((2*n_node, 2*n_node))

    for e in range(1, n_elem+1):
        for i in range(1, 3+1):
            for j in range(1, 3+1):
                for dl in range(1, 2+1):
                    for dm in range(1, 2+1):
                        L = 2 * (elems[e,i-1]-1) + dl
                        M = 2 * (elems[e,j-1]-1) + dm
                        l = 2 * (i-1) + dl
                        m = 2 * (j-1) + dm
                        K[L-1, M-1] += k[e-1, l-1, m-1]

        # progress
        # status = "{:.1f}%".format(e / float(n_elem) * 100)
        # sys.stdout.write('\r' + ' ' * len(status) + '\r')
        # sys.stdout.write(status)
        # sys.stdout.flush()

    return K

def force_vector(nodes, n_node, load_x, f_x, f_y):
    """
    力のベクトルを作る
    ---------------------
    nodes  : 節点
    n_node : 節点数
    load_x : 荷重点のx座標
    f_x    : x方向の荷重
    f_y    : y方向の荷重
    """
    # 力を格納する用
    f = np.zeros(n_node*2)

    # 荷重点の節点番号を取得(1~)
    load_idx = np.where(nodes[:, 0]==load_x)[0]

    for i in (load_idx-1):  # 0~6843-1
        f[2*i] = f_x  # x方向の力(0,2,4,6,...)
        f[2*i+1] = f_y  # y方向の力(1,3,5,7,...)
    
    return f

def set_dirichlet_boundary(K, fixed_x):
    """
    ディリクレ境界条件を設定する
    -----------------------------
    K       : 全体剛性マトリクス
    fixed_x : 固定端のx座標
    """
    # 固定端の節点番号
    fixed_idx = np.where(nodes[:, 0]==fixed_x)[0]
    for i in fixed_idx:
        j = 2*(i-1)
        K[j:j+2, :] = 0        # 行(xy)
        K[:, j:j+2] = 0        # 列(xy)
        K[[j,j+1], [j,j+1]] = 1
    
    return K

@progress_msg("solving...")
def solve(K, force):
    u = np.linalg.solve(K, force)
    return u

def save_result(elems, x_coords, y_coords, colors, f_name):
    """
    変形前と変形後の形状を画像として保存する
    -----------------------------------------------
    elems     : 全要素(ダミーを含む)
    x_coords  : x座標のリストのリスト [[x0,x1,..], [x0,x1,...], []]
    y_coords  : y座標のリストのリスト [[y0,y1,..], [y0,y1,...], []]
    colors    : グラフの色リスト
    f_name    : 保存する画像の名前 
    """
    fig = plt.figure(figsize=(10, 8)) # 横幅, 立幅
    ax = fig.add_subplot(111)         # 行数, 列数, それらのどこに配置するか

    # メッシュを描く
    for x, y, color in zip(x_coords, y_coords, colors):

        # 三角形の節点idxを取得
        triangle_idx = elems[1:,[0,1,2,0]].reshape(-1) 
        # x, yから座標を取り出す
        ax.plot(x[triangle_idx-1], y[triangle_idx-1], color=color, lw=0.2)

    fig.savefig(f_name)


if __name__ == "__main__":
    ############# 各値の設定 ##############
    POISSON = 0.3       # ポアソン比
    T = 1e-3            # 厚さ
    E = 203e9           # ヤング率
    F_X = 0             # x方向荷重
    F_Y = -1e4          # y方向荷重
    LOAD_POINT_X = 1    # 荷重点のx座標
    FIXED_X = -1        # 固定端のx座標
    DATA_FILE_NAME = "FEM_Data.dat" # データのファイル名
    #######################################

    # ファイルの読み込み
    n_node, n_elem, nodes, elems = read_fem_data(DATA_FILE_NAME)

    # 要素ごとに必要な値を求める
    k_elem_mat_list = [] # 要素剛性マトリクスを格納するためのリスト
    for i in range(n_elem):
        S = get_elem_area(nodes, elems[i+1])      # 要素の面積
        B_mat = B_matrix(nodes, elems[i+1])       # B_matrix
        D_mat = D_matrix(E, POISSON)              # D_matrix
        k_elem_mat = k_elem_matrix(B_mat, D_mat, S, T) # 要素剛性マトリクス
        k_elem_mat_list.append(k_elem_mat)
    
    # 全体剛性マトリクスを求める
    if load_k:
        K = np.load("K_mat.npy")
    else:
        K = K_matrix(np.stack(k_elem_mat_list), n_node, n_elem, nodes, elems)
        if save_k:
            np.save("K_mat", K)

    # 力の設定
    f = force_vector(nodes, n_node, load_x=LOAD_POINT_X, f_x=F_X, f_y=F_Y)
    f = f.reshape((f.shape[0], 1))

    # 境界条件を設定
    K = set_dirichlet_boundary(K, fixed_x=FIXED_X)
    
    # 変位を求める
    if load_u:
        u = np.load("u.npy")
    else:
        u = solve(K, f).reshape(-1) # ベクトルに変換
        if save_u:
            np.save("u", u)

    # 節点のx,y方向の変位
    u_x = u[np.arange(n_node*2)%2==0]
    u_y = u[np.arange(n_node*2)%2==1]
    
    # 節点のx,y方向の初期位置
    x = nodes[1:, 0]
    y = nodes[1:, 1]
     
    # 変化前と変化後の物体形状を表示
    img_name = "result.png"
    colors = ["#ff7f00", "#377eb8"]
    save_result(elems, [x, x+u_x], [y, y+u_y], colors, img_name)

    # 画像の表示
    if show_img:
        print("press q to close image viewer")
        img = cv2.imread(img_name)
        cv2.imshow(img_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

