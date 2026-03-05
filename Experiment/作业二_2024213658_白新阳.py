import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#1.加载数据&预处理
data = load_breast_cancer()
X, y = data.data, data.target          # X: (569,30)  y: (569,)
print(f"[数据] X.shape={X.shape}, y.shape={y.shape}")
print(f"[数据] 类别分布  0(恶性):{(y==0).sum()}  1(良性):{(y==1).sum()}")

# 划分训练 / 测试集（8:2）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 标准化（仅用训练集 fit，避免数据泄漏）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"[划分] 训练集:{X_train.shape}  测试集:{X_test.shape}\n")

#2.sigmoid函数
def sigmoid(z):
    """
    σ(z) = 1 / (1 + e^{-z})
    数值稳定写法：对正负值分别处理，避免 overflow
    """
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z))
    )

print(f"[验证] sigmoid(0) = {sigmoid(0)}")          # 期望输出 0.5
print(f"[验证] sigmoid(+∞)≈{sigmoid(1e9):.6f}")
print(f"[验证] sigmoid(-∞)≈{sigmoid(-1e9):.6f}\n")

#3.预测概率p = σ(Xw + b)
def predict_proba(X, w, b):
    """
    参数
    ----
    X : (n, d)  特征矩阵
    w : (d,)    权重向量
    b : float   偏置

    返回
    ----
    p : (n,)    每个样本属于正类的概率
    """
    z = X @ w + b          # 线性组合  (n,)
    return sigmoid(z)      # 映射到 (0,1)

#4.二元交叉熵损失（Log Loss）
def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    L = -1/n * Σ [y·log(p) + (1-y)·log(1-p)]

    参数
    ----
    y_true : (n,)  真实标签 {0,1}
    y_pred : (n,)  预测概率  ∈(0,1)
    eps    : float 防止 log(0) 的数值截断

    返回
    ----
    loss : float  标量损失值
    """
    p = np.clip(y_pred, eps, 1 - eps)
    loss = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    return loss

# 5. 梯度计算
def compute_gradients(X, y_true, y_pred):
    """
    ∂L/∂w = 1/n · Xᵀ(p - y)
    ∂L/∂b = 1/n · Σ(p - y)

    参数
    ----
    X      : (n, d)
    y_true : (n,)
    y_pred : (n,)  p = σ(Xw+b)

    返回
    ----
    dw : (d,)
    db : float
    """
    n = X.shape[0]
    error = y_pred - y_true          # (n,)  残差
    dw = (X.T @ error) / n           # (d,)
    db = error.mean()                # scalar
    return dw, db