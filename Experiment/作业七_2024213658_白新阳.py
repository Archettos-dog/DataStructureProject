import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import random
import os

CORPUS_PATH = "tiny_corpus_rnn.txt"   # 数据文件路径
T           = 32      # 序列长度（time steps）
E           = 32      # Embedding 维度
H           = 128     # 隐藏层维度
B           = 64      # Batch size
EPOCHS      = 30      # 训练轮数（每轮随机采若干 batch）
STEPS_PER_EPOCH = 200 # 每轮步数
LR          = 1e-3    # 学习率
GRAD_CLIP   = 5.0     # 梯度裁剪阈值
SEED        = 42

torch.manual_seed(SEED)
random.seed(SEED)

#1.数据读取与字符编码
print("=" * 60)
print("【1】数据读取与字符编码")
print("=" * 60)

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    TEXT = f.read()

vocab = sorted(set(TEXT))
vocab_size = len(vocab)          # V
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}

ids = torch.tensor([stoi[ch] for ch in TEXT], dtype=torch.long)  # (L,)

print(f"文本总长度  len(TEXT)  = {len(TEXT)}")
print(f"词表大小    vocab_size = {vocab_size}")
print(f"文本预览（前 200 字符）:\n{TEXT[:200]}\n")


def get_batch(batch_size=B, seq_len=T):
    """随机采样一个 mini-batch，返回 (x, y)，shape 均为 (B, T)"""
    starts = torch.randint(0, len(ids) - seq_len - 1, (batch_size,))
    x = torch.stack([ids[s: s + seq_len]     for s in starts])  # (B, T)
    y = torch.stack([ids[s + 1: s + seq_len + 1] for s in starts])  # (B, T)
    return x, y

#2.embedding 层
print("=" * 60)
print("【2】Embedding 层")
print("=" * 60)

embedding = nn.Embedding(vocab_size, E)

x_sample, _ = get_batch()
print(f"随机 batch 输入 shape:  x = {x_sample.shape}  (B={B}, T={T})")
x_emb_sample = embedding(x_sample)
print(f"Embedding 输出 shape:   x_emb = {x_emb_sample.shape}  (B={B}, T={T}, E={E})\n")