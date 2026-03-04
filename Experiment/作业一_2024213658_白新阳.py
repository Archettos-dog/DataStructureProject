import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

#0.可复现性
torch.manual_seed(42)

#1.载入数据&划分&标准化
print("=" * 50)
print("Step 1 — 数据载入与预处理")
print("=" * 50)

X, y = load_digits(return_X_y=True)          # X: (1797, 64), y: (1797,)

# 先划出 test (20%)，再把剩余部分划出 val (12.5% → 整体约 10%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.125, random_state=42, stratify=y_train_val
)

# StandardScaler（用 train 集 fit）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

print(f"X_train: {X_train.shape},  y_train: {y_train.shape}")
print(f"X_val  : {X_val.shape},    y_val  : {y_val.shape}")
print(f"X_test : {X_test.shape},   y_test : {y_test.shape}")