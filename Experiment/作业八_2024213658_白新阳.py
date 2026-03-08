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

# 4. BoW + Softmax 分类器（从零实现，使用 PyTorch）
print("\n" + "=" * 60)
print("4. BoW + Softmax 分类器（Adam 优化）")
print("=" * 60)

num_classes = len(CATEGORIES)

# 转为 Tensor
X_tr = torch.tensor(X_train_bow)
y_tr = torch.tensor(labels_train, dtype=torch.long)
X_vl = torch.tensor(X_val_bow)
y_vl = torch.tensor(labels_val, dtype=torch.long)
X_te = torch.tensor(X_test_bow)
y_te = torch.tensor(labels_test, dtype=torch.long)

# 模型参数
W_bow = nn.Parameter(torch.randn(vocab_size, num_classes) * 0.01)
b_bow = nn.Parameter(torch.zeros(num_classes))
optimizer_bow = optim.Adam([W_bow, b_bow], lr=LR)
ce_loss = nn.CrossEntropyLoss()

train_dataset = TensorDataset(X_tr, y_tr)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

def eval_acc(X, y):
    with torch.no_grad():
        logits = X @ W_bow + b_bow
        preds = logits.argmax(dim=1)
        return (preds == y).float().mean().item()

for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    for xb, yb in train_loader:
        logits = xb @ W_bow + b_bow  # (B, C)
        loss = ce_loss(logits, yb)
        optimizer_bow.zero_grad()
        loss.backward()
        optimizer_bow.step()
        total_loss += loss.item() * len(xb)
    avg_loss = total_loss / len(X_tr)
    val_acc  = eval_acc(X_vl, y_vl)
    print(f"  Epoch {epoch:2d}/{EPOCHS} | train loss: {avg_loss:.4f} | val acc: {val_acc:.4f}")

test_acc_bow = eval_acc(X_te, y_te)
print(f"\n[BoW + Softmax] Test Accuracy: {test_acc_bow:.4f}")

# 5. BoW 可解释性分析
print("\n" + "=" * 60)
print("5. BoW 可解释性分析（每类 Top-10 词）")
print("=" * 60)

W_np = W_bow.detach().numpy()  # (V, C)
for c, cat_name in enumerate(CATEGORIES):
    top10_ids = np.argsort(W_np[:, c])[::-1][:10]
    top10_words = [vocab_words[i] for i in top10_ids]
    print(f"\n  类别 {c} ({cat_name}):")
    print(f"    {top10_words}")

# 6. Embedding 平均池化表示 + 分类
print("\n" + "=" * 60)
print("6. Embedding 平均池化（EmbeddingBag）分类器")
print("=" * 60)

# 编码所有文本为 id 列表
def encode_all(texts_list):
    return [encode(t) for t in texts_list]

ids_train = encode_all(texts_train)
ids_val   = encode_all(texts_val)
ids_test  = encode_all(texts_test)

def pad_batch(id_lists, pad_id=0):
    """将变长 id 列表 pad 到同长度，返回 (ids_tensor, mask_tensor)"""
    max_len = max(max((len(s) for s in id_lists), default=1), 1)
    B = len(id_lists)
    ids_arr  = np.zeros((B, max_len), dtype=np.int64)
    mask_arr = np.zeros((B, max_len), dtype=np.float32)
    for i, ids in enumerate(id_lists):
        L = len(ids)
        if L > 0:
            ids_arr[i, :L]  = ids
            mask_arr[i, :L] = 1.0
    return torch.tensor(ids_arr), torch.tensor(mask_arr)

class EmbeddingAvgClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, ids, mask):
        """
        ids:  (B, L) LongTensor
        mask: (B, L) FloatTensor，1 表示有效位置
        """
        emb = self.embedding(ids)           # (B, L, E)
        # masked average pooling
        mask_exp = mask.unsqueeze(-1)       # (B, L, 1)
        sum_emb  = (emb * mask_exp).sum(1)  # (B, E)
        lengths  = mask_exp.sum(1).clamp(min=1)  # (B, 1)
        e_bar    = sum_emb / lengths        # (B, E)
        logits   = self.fc(e_bar)           # (B, C)
        return logits

model_emb = EmbeddingAvgClassifier(vocab_size, EMBED_DIM, num_classes)
optimizer_emb = optim.Adam(model_emb.parameters(), lr=LR)

y_tr_t = torch.tensor(labels_train, dtype=torch.long)
y_vl_t = torch.tensor(labels_val,   dtype=torch.long)
y_te_t = torch.tensor(labels_test,  dtype=torch.long)

# 构建整批 DataLoader（每个 epoch 动态 pad）
class IdDataset(torch.utils.data.Dataset):
    def __init__(self, id_lists, labels):
        self.id_lists = id_lists
        self.labels   = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.id_lists[idx], self.labels[idx]

def collate_fn(batch):
    id_lists, labels = zip(*batch)
    ids_t, mask_t = pad_batch(id_lists)
    return ids_t, mask_t, torch.tensor(labels, dtype=torch.long)

train_ds_emb = IdDataset(ids_train, labels_train)
train_dl_emb = DataLoader(train_ds_emb, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

def eval_acc_emb(id_lists, y_tensor):
    model_emb.eval()
    with torch.no_grad():
        ids_t, mask_t = pad_batch(id_lists)
        logits = model_emb(ids_t, mask_t)
        preds  = logits.argmax(dim=1)
        return (preds == y_tensor).float().mean().item()

for epoch in range(1, EPOCHS + 1):
    model_emb.train()
    total_loss = 0.0
    for ids_b, mask_b, yb in train_dl_emb:
        logits = model_emb(ids_b, mask_b)
        loss   = ce_loss(logits, yb)
        optimizer_emb.zero_grad()
        loss.backward()
        optimizer_emb.step()
        total_loss += loss.item() * len(yb)
    avg_loss = total_loss / len(ids_train)
    val_acc  = eval_acc_emb(ids_val, y_vl_t)
    print(f"  Epoch {epoch:2d}/{EPOCHS} | train loss: {avg_loss:.4f} | val acc: {val_acc:.4f}")

test_acc_emb = eval_acc_emb(ids_test, y_te_t)
print(f"\n[Embedding Avg + Linear] Test Accuracy: {test_acc_emb:.4f}")

# 汇总
print("\n" + "=" * 60)
print("实验结果汇总")
print("=" * 60)
print(f"  BoW + Softmax      Test Acc: {test_acc_bow:.4f}")
print(f"  Embedding Avg+FC   Test Acc: {test_acc_emb:.4f}")
print("=" * 60)