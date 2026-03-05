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