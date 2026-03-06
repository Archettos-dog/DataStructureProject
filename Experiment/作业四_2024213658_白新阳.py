import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#1.数据准备
digits = load_digits()
X, y = digits.data, digits.target          # (1797, 64), (1797,)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

N_train, D = X_train.shape   # D = 64
H = 128                       # 隐藏层维度
C = 10                        # 类别数

#2.工具函数
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def softmax(x):
    x = x - x.max(axis=1, keepdims=True)   # 数值稳定
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy_loss(probs, y):
    """平均交叉熵损失"""
    N = len(y)
    log_probs = -np.log(probs[np.arange(N), y] + 1e-12)
    return log_probs.mean()

def accuracy(probs, y):
    return (probs.argmax(axis=1) == y).mean()

def one_hot(y, C):
    N = len(y)
    oh = np.zeros((N, C))
    oh[np.arange(N), y] = 1.0
    return oh

#3.Dropout层（Inverted Dropout）
class DropoutLayer:
    """
    Inverted Dropout:
      - 训练时：以概率 p 随机置零，并除以 (1-p) 保持期望不变
      - 测试时：直接透传，无需任何缩放
    """
    def __init__(self, p=0.5):
        assert 0.0 <= p < 1.0, "丢弃概率 p 须在 [0, 1) 内"
        self.p = p          # 丢弃概率
        self.mask = None
        self.training = True

    def train_mode(self):
        self.training = True

    def eval_mode(self):
        self.training = False

    def forward(self, x):
        if self.training and self.p > 0:
            # 生成 Bernoulli 掩码，保留概率为 (1-p)
            self.mask = (np.random.rand(*x.shape) > self.p).astype(float)
            return x * self.mask / (1.0 - self.p)   # 缩放保持期望
        else:
            self.mask = None
            return x   # eval 模式：直接返回

    def backward(self, dout):
        if self.training and self.p > 0:
            return dout * self.mask / (1.0 - self.p)
        return dout
    
#4.MLP
class MLP:
    """
    结构：Linear(W1,b1) → ReLU → Dropout(p) → Linear(W2,b2) → Softmax
    """
    def __init__(self, D, H, C, dropout_p=0.0, seed=0):
        rng = np.random.RandomState(seed)
        # He 初始化
        self.W1 = rng.randn(D, H) * np.sqrt(2.0 / D)
        self.b1 = np.zeros(H)
        self.W2 = rng.randn(H, C) * np.sqrt(2.0 / H)
        self.b2 = np.zeros(C)

        self.dropout = DropoutLayer(p=dropout_p)
        self.training = True

        # 缓存前向中间值（用于反向传播）
        self._cache = {}

    def train_mode(self):
        self.training = True
        self.dropout.train_mode()

    def eval_mode(self):
        self.training = False
        self.dropout.eval_mode()

    def forward(self, X):
        # 第一层
        z1 = X @ self.W1 + self.b1          # (N, H)
        a1 = relu(z1)                        # (N, H)
        a1_drop = self.dropout.forward(a1)   # (N, H)  Dropout

        # 第二层
        z2 = a1_drop @ self.W2 + self.b2    # (N, C)
        probs = softmax(z2)                  # (N, C)

        self._cache = dict(X=X, z1=z1, a1=a1, a1_drop=a1_drop, z2=z2, probs=probs)
        return probs

    def backward(self, y):
        """反向传播，返回梯度字典"""
        cache = self._cache
        N = len(y)
        probs = cache['probs']

        # Softmax + 交叉熵 梯度
        dz2 = probs - one_hot(y, C)          # (N, C)
        dz2 /= N

        dW2 = cache['a1_drop'].T @ dz2       # (H, C)
        db2 = dz2.sum(axis=0)                # (C,)

        da1_drop = dz2 @ self.W2.T           # (N, H)
        da1 = self.dropout.backward(da1_drop)

        dz1 = da1 * relu_grad(cache['z1'])   # (N, H)
        dW1 = cache['X'].T @ dz1             # (D, H)
        db1 = dz1.sum(axis=0)                # (H,)

        return dict(W1=dW1, b1=db1, W2=dW2, b2=db2)

    def params(self):
        return dict(W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)