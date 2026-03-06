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