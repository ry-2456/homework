import numpy as np

class BaseLayer:

    def __init__(self, n_upper, n, eta, alpha):
        """ n_upper: 上の層のニューロン数
            n      : この層のニューロン数 """
        self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper) # 重み(Xavier)
        self.b = np.zeros(n)                                      # バイアス

        self.eta = eta                                            # 学習係数
        self.alpha = alpha                                        # 安定化係数

        self.prev_update_amt_w = np.zeros((n_upper, n))           # 前回のwの更新量
        self.prev_update_amt_b = np.zeros(n)                      # 前回のbの更新量
    
    def update(self):
        """momentum"""
        # 更新量の計算
        update_amt_w = self.eta * self.grad_w + self.alpha * self.prev_update_amt_w 
        update_amt_b = self.eta * self.grad_b + self.alpha * self.prev_update_amt_b

        self.w -= update_amt_w
        self.b -= update_amt_b

        self.prev_update_amt_w = update_amt_w
        self.prev_update_amt_b = update_amt_b


class MiddleLayer(BaseLayer):
    
    def forward_prop(self, x):
        """sigmodi関数を使用"""
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = 1 / (1 + np.exp(-u)) # sigmoid

    def back_prop(self, grad_y):
        delta = grad_y * (1 - self.y) * self.y # sigmoidの微分

        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)

        self.grad_x = np.dot(delta, self.w.T)


class OutputLayer(BaseLayer):
    
    def forward_prop(self, x):
        """ソフトマックス関数を使用"""
        self.x = x 
        u = np.dot(x, self.w) + self.b
        self.y = np.exp(u) / np.sum(np.exp(u), axis=1, keepdims=True)

    def back_prop(self, t):
        delta = self.y - t

        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)

        self.grad_x = np.dot(delta, self.w.T)

class NeuralNetwork:
    
    def __init__(self, layers):
        """中間層1つと出力層1つを想定"""
        self.middle_layer = layers[0]
        self.output_layer = layers[1]

    # 順伝播
    def forward_prop(self, input_):
        self.middle_layer.forward_prop(input_)
        self.output_layer.forward_prop(self.middle_layer.y)

    # 逆伝播
    def back_prop(self, correct):
        self.output_layer.back_prop(correct)
        self.middle_layer.back_prop(self.output_layer.grad_x)

    # 重みとバイアスの更新
    def update_wb(self):
        self.middle_layer.update()
        self.output_layer.update()

    # クロスエントロピー誤差
    def cross_entropy(self, correct, batch_size):
        return -np.sum(correct * np.log(self.output_layer.y + 1e-7)) / batch_size

    # 正答率
    def accuracy(self, input_, correct):
        self.middle_layer.forward_prop(input_)
        self.output_layer.forward_prop(self.middle_layer.y)

        cnt_correct = np.sum(np.argmax(self.output_layer.y, axis=1) == np.argmax(correct, axis=1))
        return cnt_correct / input_.shape[0] * 100
