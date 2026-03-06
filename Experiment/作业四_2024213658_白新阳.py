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

