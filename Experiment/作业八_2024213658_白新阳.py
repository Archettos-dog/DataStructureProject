"""
实验：词袋模型（BoW）与 Embedding 平均池化文本分类
数据：优先尝试 sklearn 20Newsgroups；若下载失败则自动切换到本地合成数据。
"""

import re
import random
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 0. 超参数与随机种子
SEED = 42
random.seed(SEED)
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

# ---------- 合成数据（网络不可用时的离线替代） ----------
_SEED_WORDS = {
    0: [  # rec.autos
        "car", "engine", "vehicle", "auto", "driver", "speed", "brake",
        "tire", "fuel", "transmission", "sedan", "suv", "turbo", "gear",
        "horsepower", "exhaust", "racing", "highway", "dealer", "clutch",
        "carburetor", "suspension", "axle", "torque", "mileage", "diesel",
        "hybrid", "electric", "battery", "motor", "wheel", "steering",
        "bumper", "hood", "trunk", "windshield", "dashboard", "oil",
        "filter", "coolant", "radiator", "spark", "ignition", "drive",
    ],
    1: [  # sci.space
        "space", "nasa", "orbit", "rocket", "satellite", "moon", "mars",
        "planet", "star", "galaxy", "telescope", "astronaut", "launch",
        "shuttle", "gravity", "solar", "cosmic", "universe", "probe",
        "mission", "asteroid", "comet", "nebula", "atmosphere", "altitude",
        "payload", "spacecraft", "module", "station", "hubble", "voyager",
        "apollo", "capsule", "thruster", "trajectory", "reentry", "lander",
        "rover", "crater", "eclipse", "observatory", "spectrum", "photon",
    ],
    2: [  # comp.graphics
        "graphics", "pixel", "render", "image", "opengl", "shader", "texture",
        "polygon", "vertex", "mesh", "raytracing", "gpu", "frame", "buffer",
        "resolution", "color", "bitmap", "vector", "animation", "display",
        "rasterize", "pipeline", "matrix", "transform", "normal", "lighting",
        "shadow", "antialiasing", "compression", "jpeg", "png", "filter",
        "kernel", "convolution", "edge", "segmentation", "depth", "blend",
        "alpha", "canvas", "viewport", "clipping", "frustum", "camera",
    ],
    3: [  # talk.politics.misc
        "politics", "government", "election", "policy", "senator", "vote",
        "democrat", "republican", "congress", "president", "law", "bill",
        "debate", "campaign", "tax", "budget", "welfare", "reform", "party",
        "civil", "rights", "freedom", "constitution", "federal", "state",
        "immigration", "healthcare", "abortion", "gun", "regulation",
        "lobby", "media", "propaganda", "corruption", "democracy", "liberal",
        "conservative", "ideology", "protest", "legislation", "supreme",
        "judiciary", "amendment", "ballot", "referendum", "candidate",
    ],
}

_COMMON_WORDS = [
    "the", "a", "is", "in", "of", "and", "to", "that", "it", "with",
    "for", "on", "are", "this", "was", "have", "from", "at", "by", "be",
    "as", "we", "they", "but", "not", "he", "she", "which", "can", "more",
    "would", "about", "also", "one", "some", "their", "when", "there", "been",
]

def _make_doc(class_id, rng, min_words=40, max_words=120):
    n = rng.randint(min_words, max_words + 1)
    seed_pool = _SEED_WORDS[class_id]
    words = []
    for _ in range(n):
        if rng.random() < 0.40:
            words.append(rng.choice(seed_pool))
        else:
            words.append(rng.choice(_COMMON_WORDS))
    return " ".join(words)

def _build_synthetic_data(samples_per_class=500, seed=42):
    rng = random.Random(seed)
    texts, labels = [], []
    for c in range(len(CATEGORIES)):
        for _ in range(samples_per_class):
            texts.append(_make_doc(c, rng))
            labels.append(c)
    return texts, np.array(labels)

# ---------- 直接从本地 tar.gz 解析，绕过 sklearn 校验 ----------
import tarfile, os

LOCAL_TAR = r"C:\Users\AYIES\scikit_learn_data\20news_home\20news-bydate.tar.gz"

def _load_from_tar(tar_path, categories):
    """
    直接解析 20news-bydate.tar.gz，返回 (texts, labels)。
    支持两种常见目录结构：
      20news-bydate-train/<category>/<file>
      20_newsgroups/<category>/<file>
    """
    cat_set = set(categories)
    cat2id  = {c: i for i, c in enumerate(categories)}
    texts, labels = [], []

    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        for m in members:
            if not m.isfile():
                continue
            parts = m.name.replace("\\", "/").split("/")
            # 找到 category 层：parts[-2] 是类别，parts[-1] 是文件
            if len(parts) < 2:
                continue
            cat = parts[-2]
            if cat not in cat_set:
                continue
            f = tar.extractfile(m)
            if f is None:
                continue
            try:
                raw = f.read().decode("utf-8", errors="replace")
            except Exception:
                continue
            # 去除 header（第一个空行之前的内容）
            if "\n\n" in raw:
                raw = raw[raw.index("\n\n") + 2:]
            texts.append(raw)
            labels.append(cat2id[cat])

    return texts, np.array(labels)

texts_all, labels_all = None, None
if os.path.exists(LOCAL_TAR):
    try:
        print(f"正在读取本地数据文件：{LOCAL_TAR}")
        texts_all, labels_all = _load_from_tar(LOCAL_TAR, CATEGORIES)
        print(f"✓ 成功加载真实数据，共 {len(texts_all)} 条。")
    except Exception as e:
        print(f"✗ 读取本地文件失败（{e}），切换到合成数据。")
        texts_all, labels_all = _build_synthetic_data(SAMPLES_PER_CLASS, SEED)
else:
    print(f"✗ 未找到本地文件 {LOCAL_TAR}，切换到合成数据。")
    texts_all, labels_all = _build_synthetic_data(SAMPLES_PER_CLASS, SEED)

# ---------- 每类采样 SAMPLES_PER_CLASS ----------
indices = []
for c in range(len(CATEGORIES)):
    idx = np.where(labels_all == c)[0]
    n   = min(SAMPLES_PER_CLASS, len(idx))
    idx = np.random.choice(idx, size=n, replace=False)
    indices.append(idx)
indices = np.concatenate(indices)
np.random.shuffle(indices)

texts  = [texts_all[i] for i in indices]
labels = labels_all[indices]

total   = len(texts)
n_train = 1600
n_val   = 200
n_test  = total - n_train - n_val

texts_train, labels_train = texts[:n_train],              labels[:n_train]
texts_val,   labels_val   = texts[n_train:n_train+n_val],  labels[n_train:n_train+n_val]
texts_test,  labels_test  = texts[n_train+n_val:],        labels[n_train+n_val:]

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
    """最简 tokenizer：小写 → 非字母数字替换空格 → 合并空格 → split"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

word_counter = Counter()
for t in texts_train:
    word_counter.update(tokenize(t))

vocab_words = ["<unk>"] + [w for w, _ in word_counter.most_common(TOP_K)]
word2id     = {w: i for i, w in enumerate(vocab_words)}
vocab_size  = len(vocab_words)
UNK_ID      = 0

print(f"\nvocab_size (含 <unk>): {vocab_size}")

def encode(text: str):
    return [word2id.get(tok, UNK_ID) for tok in tokenize(text)]

sample_tokens = tokenize(texts_train[0])[:15]
sample_ids    = [word2id.get(t, UNK_ID) for t in sample_tokens]
print(f"\n示例文本前 15 个 token: {sample_tokens}")
print(f"对应词 id:              {sample_ids}")

# 3. 实现 BoW 特征（简化 TF-IDF + L2 归一化）
print("\n" + "=" * 60)
print("3. 实现 BoW 特征（简化 TF-IDF + L2 归一化）")
print("=" * 60)

N_train = len(texts_train)
df = np.zeros(vocab_size, dtype=np.float32)
for t in texts_train:
    for i in set(encode(t)):
        df[i] += 1

idf = np.log((N_train + 1) / (df + 1)) + 1   # 平滑 IDF

def text_to_bow(text: str) -> np.ndarray:
    ids = encode(text)
    if not ids:
        return np.zeros(vocab_size, dtype=np.float32)
    tf = np.zeros(vocab_size, dtype=np.float32)
    for i in ids:
        tf[i] += 1
    tf  /= len(ids)
    vec  = tf * idf
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec

X_train_bow = np.stack([text_to_bow(t) for t in texts_train]).astype(np.float32)
X_val_bow   = np.stack([text_to_bow(t) for t in texts_val  ]).astype(np.float32)
X_test_bow  = np.stack([text_to_bow(t) for t in texts_test ]).astype(np.float32)

print(f"\nX_train_bow 维度: {X_train_bow.shape}")
print(f"X_val_bow   维度: {X_val_bow.shape}")
print(f"X_test_bow  维度: {X_test_bow.shape}")

# 4. BoW + Softmax 分类器（从零实现）
print("\n" + "=" * 60)
print("4. BoW + Softmax 分类器（logits = x_bow @ W + b，Adam 优化）")
print("=" * 60)

num_classes = len(CATEGORIES)

X_tr = torch.tensor(X_train_bow)
y_tr = torch.tensor(labels_train, dtype=torch.long)
X_vl = torch.tensor(X_val_bow)
y_vl = torch.tensor(labels_val,   dtype=torch.long)
X_te = torch.tensor(X_test_bow)
y_te = torch.tensor(labels_test,  dtype=torch.long)

W_bow = nn.Parameter(torch.randn(vocab_size, num_classes) * 0.01)
b_bow = nn.Parameter(torch.zeros(num_classes))
optimizer_bow = optim.Adam([W_bow, b_bow], lr=LR)
ce_loss = nn.CrossEntropyLoss()

class SimpleDataset(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self):        return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(SimpleDataset(X_tr, y_tr),
                          batch_size=BATCH_SIZE, shuffle=True)

def eval_acc_bow(X, y):
    with torch.no_grad():
        preds = (X @ W_bow + b_bow).argmax(dim=1)
        return (preds == y).float().mean().item()

for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    for xb, yb in train_loader:
        logits = xb @ W_bow + b_bow
        loss   = ce_loss(logits, yb)
        optimizer_bow.zero_grad()
        loss.backward()
        optimizer_bow.step()
        total_loss += loss.item() * len(xb)
    avg_loss = total_loss / N_train
    val_acc  = eval_acc_bow(X_vl, y_vl)
    print(f"  Epoch {epoch:2d}/{EPOCHS} | train loss: {avg_loss:.4f} | val acc: {val_acc:.4f}")

test_acc_bow = eval_acc_bow(X_te, y_te)
print(f"\n[BoW + Softmax] Test Accuracy: {test_acc_bow:.4f}")

# 5. BoW 可解释性分析
print("\n" + "=" * 60)
print("5. BoW 可解释性分析（每类 Top-10 词）")
print("=" * 60)

W_np = W_bow.detach().numpy()
for c, cat_name in enumerate(CATEGORIES):
    top10_ids   = np.argsort(W_np[:, c])[::-1][:10]
    top10_words = [vocab_words[i] for i in top10_ids]
    print(f"\n  类别 {c} ({cat_name}):")
    print(f"    {top10_words}")

# 6. Embedding 平均池化表示 + 分类
print("\n" + "=" * 60)
print("6. Embedding 平均池化（EmbeddingBag）分类器")
print("=" * 60)

ids_train = [encode(t) for t in texts_train]
ids_val   = [encode(t) for t in texts_val  ]
ids_test  = [encode(t) for t in texts_test ]

def pad_batch(id_lists):
    """变长序列 → (ids_tensor, mask_tensor)"""
    max_len = max((len(s) for s in id_lists), default=1)
    max_len = max(max_len, 1)
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
        self.fc         = nn.Linear(embed_dim, num_classes)

    def forward(self, ids, mask):
        """
        ids  : (B, L) LongTensor
        mask : (B, L) FloatTensor — 1=有效位置，0=padding
        """
        emb      = self.embedding(ids)              # (B, L, E)
        mask_exp = mask.unsqueeze(-1)               # (B, L, 1)
        sum_emb  = (emb * mask_exp).sum(dim=1)      # (B, E)
        lengths  = mask_exp.sum(dim=1).clamp(min=1) # (B, 1)
        e_bar    = sum_emb / lengths                # (B, E) masked average
        return self.fc(e_bar)                       # (B, C)

model_emb     = EmbeddingAvgClassifier(vocab_size, EMBED_DIM, num_classes)
optimizer_emb = optim.Adam(model_emb.parameters(), lr=LR)

class IdDataset(Dataset):
    def __init__(self, id_lists, labels):
        self.id_lists = id_lists
        self.labels   = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.id_lists[idx], int(self.labels[idx])

def collate_fn(batch):
    id_lists, labels = zip(*batch)
    ids_t, mask_t = pad_batch(id_lists)
    return ids_t, mask_t, torch.tensor(labels, dtype=torch.long)

train_dl_emb = DataLoader(
    IdDataset(ids_train, labels_train),
    batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)

def eval_acc_emb(id_lists, labels_np):
    model_emb.eval()
    with torch.no_grad():
        ids_t, mask_t = pad_batch(id_lists)
        logits = model_emb(ids_t, mask_t)
        preds  = logits.argmax(dim=1)
        y      = torch.tensor(labels_np, dtype=torch.long)
        return (preds == y).float().mean().item()

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
    val_acc  = eval_acc_emb(ids_val, labels_val)
    print(f"  Epoch {epoch:2d}/{EPOCHS} | train loss: {avg_loss:.4f} | val acc: {val_acc:.4f}")

test_acc_emb = eval_acc_emb(ids_test, labels_test)
print(f"\n[Embedding Avg + Linear] Test Accuracy: {test_acc_emb:.4f}")

# 汇总
print("\n" + "=" * 60)
print("实验结果汇总")
print("=" * 60)
print(f"  BoW + Softmax          Test Acc: {test_acc_bow:.4f}")
print(f"  Embedding Avg + Linear Test Acc: {test_acc_emb:.4f}")
print("=" * 60)