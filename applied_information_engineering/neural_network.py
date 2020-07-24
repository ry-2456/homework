import numpy as np

class BaseLayer:

    def __init__(self, n_upper, n, eta, alpha):
        """ n_upper: 上の層のニューロン数
            n      : この層のニューロン数 """
        self.w = np.random.randn(n_upper, n) * np.sqrt(n_upper) # 重み(Xavier)
        self.b = np.zeros(n)                                    # バイアス

        self.eta = eta                                          # 学習係数
        self.alpha = alpha                                      # 安定化係数

        self.prev_grad_w = np.zeros((n_upper, n))               # 前回のwの更新量
        self.prev_grad_b = np.zeros(n)                          # 前回のbの更新量
    
    def update(self):
        """momentum"""
        self.w -= (self.eta * self.grad_w + self.alpha * self.prev_grad_w)
        self.b -= (slef.eta * self.grad_b + slef.alpha * self.prev_grad_b)


class MiddleLayer(BaseLayer):
    
    def forward_prop(self, x):
        """sigmodi関数を使用"""
        self.x = x
        self.u = np.dot(x, self.w) + self.b
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
