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

# 2. 构建词表与分词
print("\n" + "=" * 60)
print("2. 构建词表与分词")
print("=" * 60)

def tokenize(text: str):
    """最简 tokenizer：小写 → 非字母数字替换为空格 → split"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

# 统计训练集词频
word_counter = Counter()
for t in texts_train:
    word_counter.update(tokenize(t))

# 取 top-K，加入 <unk>
vocab_words = ["<unk>"] + [w for w, _ in word_counter.most_common(TOP_K)]
word2id = {w: i for i, w in enumerate(vocab_words)}
vocab_size = len(vocab_words)
UNK_ID = 0

print(f"\nvocab_size (含 <unk>): {vocab_size}")

# 示例
sample_text = texts_train[0]
sample_tokens = tokenize(sample_text)[:15]
sample_ids = [word2id.get(tok, UNK_ID) for tok in sample_tokens]
print(f"\n示例文本前 15 个 token: {sample_tokens}")
print(f"对应词 id:              {sample_ids}")

def encode(text: str):
    return [word2id.get(tok, UNK_ID) for tok in tokenize(text)]

# 3. 实现 BoW 特征（简化 TF-IDF + L2 归一化）
print("\n" + "=" * 60)
print("3. 实现 BoW 特征（简化 TF-IDF）")
print("=" * 60)

# 计算 IDF：在训练集上统计 DF
df = np.zeros(vocab_size, dtype=np.float32)
for t in texts_train:
    ids_set = set(encode(t))
    for i in ids_set:
        df[i] += 1

N_train = len(texts_train)
idf = np.log((N_train + 1) / (df + 1)) + 1  # 平滑 IDF

def text_to_bow(text: str) -> np.ndarray:
    """返回 TF-IDF 加权的 BoW 向量，再 L2 归一化"""
    ids = encode(text)
    if len(ids) == 0:
        return np.zeros(vocab_size, dtype=np.float32)
    # TF：长度归一的词频
    tf = np.zeros(vocab_size, dtype=np.float32)
    for i in ids:
        tf[i] += 1
    tf /= len(ids)
    # TF-IDF
    vec = tf * idf
    # L2 归一化
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec

# 构建矩阵
X_train_bow = np.stack([text_to_bow(t) for t in texts_train]).astype(np.float32)
X_val_bow   = np.stack([text_to_bow(t) for t in texts_val]).astype(np.float32)
X_test_bow  = np.stack([text_to_bow(t) for t in texts_test]).astype(np.float32)

print(f"\nX_train_bow 维度: {X_train_bow.shape}")
print(f"X_val_bow   维度: {X_val_bow.shape}")
print(f"X_test_bow  维度: {X_test_bow.shape}")