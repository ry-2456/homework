import numpy as np
import sys

def read_fem_data(file_name):
    ################################################################
    ################################################################
    #                        精度に問題あり                        # 
    ################################################################
    ################################################################
    with open(file_name) as f:
        node_n = int(f.readline().split()[0]) # the number of nodes
        elem_n = int(f.readline().split()[0]) # the number of triangle
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
        # for i in range(node_n):
        #     print(nodes[i+1])
        # input("Enter to continue")

        # for i in range(elem_n):
        #     print(elems[i+1])

# def get_elem_area(nodes, elem):
#     # elemの面積を返す
#     u1, u2, u3 = elem
#     area_mat = np.ones((3, 3)) # 面積を求めるための行列
#     area_mat[:, :2] = nodes[elem, :]
# 
#     return 0.5 * np.abs(np.linalg.det(area_mat))

def get_elem_area(nodes, elem):
    # elemの面積を返す
    u1, u2, u3 = elem
    x1, y1 = nodes[u1] 
    x2, y2 = nodes[u2]
    x3, y3 = nodes[u3]
    s = np.abs(0.5 * (y1*(x2-x3) + y2*(x3-x1) + y3*(x1-x2)))
    return s

def B_matrix(nodes, elem):
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
    c = E / (1 - v ** 2) 
    D_mat = c * np.array([[1, v, 0],
                          [v, 1, 0],
                          [0, 0, (1-v)/2.0]])
    return D_mat

def k_elem_matrix(B, D, S, t):
    k = S * t * B.T.dot(D).dot(B)
    return k

def K_matrix(k, node_n, elem_n, nodes, elems):
    # for e in range(1, elem_n +1):
    K = np.zeros((2*node_n, 2*node_n))
    # print(k.shape)
    # print(K.shape)

    for e in range(1, elem_n+1):
        for i in range(1, 3+1):
            for j in range(1, 3+1):
                for dl in range(1, 2+1):
                    for dm in range(1, 2+1):
                        l = 2 * (i-1) + dl
                        m = 2 * (j-1) + dm
                        L = 2 * (elems[e,i-1]-1) + dl
                        M = 2 * (elems[e,j-1]-1) + dm
                        # print(k[e-1, l-1, m-1])
                        K[L-1, M-1] = K[L-1, M-1] + k[e-1, l-1, m-1]

        # progress
        status = "{:.1f}%".format(e / float(elem_n) * 100)
        sys.stdout.write('\r' + ' ' * len(status) + '\r')
        sys.stdout.write(status)
        sys.stdout.flush()

    # print(K)
    # print(K.shape)
    return K

if __name__ == "__main__":
    poisson_ratio = 0.3 # ポアソン比
    t = 1               # 厚さ
    E = 1               # ヤング率
    data_file_name = "FEM_data.txt"
    node_n, elem_n, nodes, elems = read_fem_data(data_file_name)
    
    k_mat_list = [] # 要素剛性マトリクスを格納するためのリスト

    for i in range(elem_n):
        S = get_elem_area(nodes, elems[i+1])
        B = B_matrix(nodes, elems[i+1])
        D = D_matrix(E, poisson_ratio)
        k = k_elem_matrix(B, D, S, t)
        k_mat_list.append(k)
        # print(k)
        # print("#################")
        # input("Enter to continue")
    
    K = K_matrix(np.stack(k_mat_list), node_n, elem_n, nodes, elems)
    print(K)
    print(K.shape)
