import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#1.数据准备
print("=" * 60)
print("1. 数据准备")
print("=" * 60)

digits = load_digits()
X, y = digits.data, digits.target
print(f"原始数据: X.shape={X.shape}, y.shape={y.shape}, 类别数={len(np.unique(y))}")

# 划分 train/val/test = 60/20/20
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)
print("标准化完成 (StandardScaler fit on train)")

#2.参数初始化
print("\n" + "=" * 60)
print("2. 参数初始化")
print("=" * 60)

np.random.seed(42)

D = 64    # 输入维度
H = 128   # 隐藏层维度
C = 10    # 输出类别数

W1 = np.random.randn(D, H) * 0.01   # (64, 128)
b1 = np.zeros(H)                     # (128,)
W2 = np.random.randn(H, C) * 0.01   # (128, 10)
b2 = np.zeros(C)                     # (10,)

print(f"W1.shape = {W1.shape}")
print(f"b1.shape = {b1.shape}")
print(f"W2.shape = {W2.shape}")
print(f"b2.shape = {b2.shape}")

#3.激活函数与工具函数
def relu(x):
    """ReLU: max(0, x)"""
    return np.maximum(0, x)

def relu_grad(x):
    """ReLU 导数：x>0 时为1，否则为0"""
    return (x > 0).astype(float)

def softmax(z):
    """数值稳定的 Softmax，输入 (B, C)，输出 (B, C)"""
    z_stable = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(P, y):
    """
    多分类交叉熵损失
    P: (B, C) softmax 输出概率
    y: (B,)   整数标签
    返回标量 loss
    """
    B = P.shape[0]
    # 取每个样本真实类别的概率，clip 防止 log(0)
    correct_probs = P[np.arange(B), y]
    loss = -np.mean(np.log(correct_probs + 1e-12))
    return loss

#4.前向传播
def forward(Xb, W1, b1, W2, b2):
    """
    前向传播
    Xb: (B, D)
    返回: P (B, C), cache (用于反向传播)
    """
    Z1 = Xb @ W1 + b1          # (B, H)
    A1 = relu(Z1)               # (B, H)
    Z2 = A1 @ W2 + b2          # (B, C)
    P  = softmax(Z2)            # (B, C)
    cache = (Xb, Z1, A1, Z2, P)
    return P, cache

#5.反向传播
def backward(cache, y, W2):
    """
    手写反向传播
    cache: (Xb, Z1, A1, Z2, P)
    y:     (B,) 整数标签
    返回: dW1, db1, dW2, db2
    """
    Xb, Z1, A1, Z2, P = cache
    B = Xb.shape[0]

    # ── 输出层梯度 ──────────────────────────────
    # dL/dZ2: softmax + cross-entropy 组合求导
    # dL/dZ2[i,j] = P[i,j] - 1(j==y[i])
    dZ2 = P.copy()                             # (B, C)
    dZ2[np.arange(B), y] -= 1
    dZ2 /= B

    # dL/dW2 = A1^T @ dZ2
    dW2 = A1.T @ dZ2                           # (H, C)
    # dL/db2 = sum over batch
    db2 = np.sum(dZ2, axis=0)                  # (C,)

    # ── 隐藏层梯度 ─────────────────────────────
    # 反传到 A1
    dA1 = dZ2 @ W2.T                           # (B, H)
    # 经过 ReLU 反传
    dZ1 = dA1 * relu_grad(Z1)                  # (B, H)

    # dL/dW1 = Xb^T @ dZ1
    dW1 = Xb.T @ dZ1                           # (D, H)
    # dL/db1 = sum over batch
    db1 = np.sum(dZ1, axis=0)                  # (H,)

    return dW1, db1, dW2, db2

# 验证梯度 shape（用一个小 batch 测试）
print("\n" + "=" * 60)
print("3. 验证前向 & 反向传播 shape")
print("=" * 60)
_Xb = X_train[:8]
_yb = y_train[:8]
_P, _cache = forward(_Xb, W1, b1, W2, b2)
print(f"输入 Xb.shape      = {_Xb.shape}")
print(f"Z1.shape           = {_cache[1].shape}")
print(f"A1.shape           = {_cache[2].shape}")
print(f"Z2.shape           = {_cache[3].shape}")
print(f"P (softmax).shape  = {_P.shape}")
print(f"loss               = {cross_entropy_loss(_P, _yb):.4f}")

_dW1, _db1, _dW2, _db2 = backward(_cache, _yb, W2)
print(f"\ndW1.shape = {_dW1.shape}")
print(f"db1.shape = {_db1.shape}")
print(f"dW2.shape = {_dW2.shape}")
print(f"db2.shape = {_db2.shape}")

#6.完整训练
print("\n" + "=" * 60)
print("4. 开始训练")
print("=" * 60)

# 超参数
lr         = 0.05
epochs     = 200
batch_size = 64

# 重新初始化参数
np.random.seed(42)
W1 = np.random.randn(D, H) * 0.01
b1 = np.zeros(H)
W2 = np.random.randn(H, C) * 0.01
b2 = np.zeros(C)

def predict(X, W1, b1, W2, b2):
    P, _ = forward(X, W1, b1, W2, b2)
    return np.argmax(P, axis=1)

def accuracy(X, y, W1, b1, W2, b2):
    preds = predict(X, W1, b1, W2, b2)
    return np.mean(preds == y)

train_losses = []
val_losses   = []
train_accs   = []
val_accs     = []

N_train = X_train.shape[0]

for epoch in range(1, epochs + 1):
    # shuffle
    idx = np.random.permutation(N_train)
    X_shuf, y_shuf = X_train[idx], y_train[idx]

    epoch_loss = 0.0
    n_batches  = 0

    for start in range(0, N_train, batch_size):
        Xb = X_shuf[start : start + batch_size]
        yb = y_shuf[start : start + batch_size]

        # 前向
        P, cache = forward(Xb, W1, b1, W2, b2)
        loss = cross_entropy_loss(P, yb)
        epoch_loss += loss
        n_batches  += 1

        # 反向
        dW1, db1_, dW2, db2_ = backward(cache, yb, W2)

        # 参数更新
        W1 -= lr * dW1
        b1 -= lr * db1_
        W2 -= lr * dW2
        b2 -= lr * db2_

    # ── 记录指标 ──────────────────────────────
    avg_loss = epoch_loss / n_batches
    train_losses.append(avg_loss)

    # val loss
    P_val, _ = forward(X_val, W1, b1, W2, b2)
    v_loss = cross_entropy_loss(P_val, y_val)
    val_losses.append(v_loss)

    t_acc = accuracy(X_train, y_train, W1, b1, W2, b2)
    v_acc = accuracy(X_val,   y_val,   W1, b1, W2, b2)
    train_accs.append(t_acc)
    val_accs.append(v_acc)

    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:>3d}/{epochs} | "
              f"Train Loss: {avg_loss:.4f} | Val Loss: {v_loss:.4f} | "
              f"Train Acc: {t_acc:.4f} | Val Acc: {v_acc:.4f}")
