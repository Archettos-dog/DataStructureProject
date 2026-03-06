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