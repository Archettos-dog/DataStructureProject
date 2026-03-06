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