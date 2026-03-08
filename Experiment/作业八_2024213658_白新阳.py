import re
import math
import numpy as np
from collections import Counter
from sklearn.datasets import fetch_20newsgroups

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 0. 超参数与随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

CATEGORIES = ["rec.autos", "sci.space", "comp.graphics", "talk.politics.misc"]
SAMPLES_PER_CLASS = 500
TOP_K = 5000
EMBED_DIM = 64
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3

# 1. 数据加载与预处理
print("=" * 60)
print("1. 数据加载与预处理")
print("=" * 60)

data = fetch_20newsgroups(
    subset="all",
    categories=CATEGORIES,
    remove=("headers", "footers", "quotes"),
    random_state=SEED,
)

texts_all = data.data
labels_all = np.array(data.target)

# 每类采样 SAMPLES_PER_CLASS
indices = []
for c in range(len(CATEGORIES)):
    idx = np.where(labels_all == c)[0]
    idx = np.random.choice(idx, size=min(SAMPLES_PER_CLASS, len(idx)), replace=False)
    indices.append(idx)
indices = np.concatenate(indices)
np.random.shuffle(indices)

texts = [texts_all[i] for i in indices]
labels = labels_all[indices]

total = len(texts)           # ~2000
n_train = 1600
n_val   = 200
n_test  = total - n_train - n_val  # 200（或剩余）

texts_train, labels_train = texts[:n_train],        labels[:n_train]
texts_val,   labels_val   = texts[n_train:n_train+n_val], labels[n_train:n_train+n_val]
texts_test,  labels_test  = texts[n_train+n_val:],  labels[n_train+n_val:]

for split_name, split_labels in [("Train", labels_train),
                                   ("Val",   labels_val),
                                   ("Test",  labels_test)]:
    print(f"\n[{split_name}] 样本数: {len(split_labels)}")
    dist = Counter(split_labels.tolist())
    for c, name in enumerate(CATEGORIES):
        print(f"  类别 {c} ({name}): {dist.get(c, 0)}")