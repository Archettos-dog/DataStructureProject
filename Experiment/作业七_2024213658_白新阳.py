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

#3.手写 MyRNNCell
print("=" * 60)
print("【3】手写 MyRNNCell")
print("=" * 60)

class MyRNNCell(nn.Module):
    """
    手写 RNN Cell（不使用 nn.RNN / nn.RNNCell）
    参数：
        Wxh : (E, H)  —— 输入到隐藏
        Whh : (H, H)  —— 隐藏到隐藏
        bh  : (H,)    —— 偏置
    前向公式：
        h_t = tanh( x_t @ Wxh + h_{t-1} @ Whh + bh )
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size  = input_size   # E
        self.hidden_size = hidden_size  # H

        # 手动定义参数（Xavier 初始化）
        self.Wxh = nn.Parameter(torch.empty(input_size, hidden_size))
        self.Whh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bh  = nn.Parameter(torch.zeros(hidden_size))

        nn.init.xavier_uniform_(self.Wxh)
        nn.init.orthogonal_(self.Whh)   # 隐藏层用正交初始化，有助于训练稳定

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        x_t   : (B, E)
        h_prev: (B, H)
        返回
        h_t   : (B, H)
        """
        # ── Shape 断言 ──────────────────────────────────────────
        assert x_t.ndim == 2 and x_t.shape[1] == self.input_size, \
            f"x_t 应为 (B, {self.input_size})，实际 {x_t.shape}"
        assert h_prev.ndim == 2 and h_prev.shape[1] == self.hidden_size, \
            f"h_prev 应为 (B, {self.hidden_size})，实际 {h_prev.shape}"
        assert x_t.shape[0] == h_prev.shape[0], \
            f"Batch size 不一致：x_t={x_t.shape[0]}，h_prev={h_prev.shape[0]}"
        # ────────────────────────────────────────────────────────

        h_t = torch.tanh(x_t @ self.Wxh + h_prev @ self.Whh + self.bh)

        # ── 输出 Shape 断言 ─────────────────────────────────────
        assert h_t.shape == h_prev.shape, \
            f"h_t shape {h_t.shape} 与 h_prev shape {h_prev.shape} 不一致"
        # ────────────────────────────────────────────────────────

        return h_t


# 快速验证 MyRNNCell
cell_test = MyRNNCell(E, H)
x_t_test  = torch.randn(B, E)
h0_test   = torch.zeros(B, H)
ht_test   = cell_test(x_t_test, h0_test)
print(f"MyRNNCell 测试：x_t={x_t_test.shape}, h_prev={h0_test.shape} → h_t={ht_test.shape}\n")

#4.时间展开
print("=" * 60)
print("【4】时间展开（Unroll）")
print("=" * 60)

def rnn_unroll(cell: MyRNNCell, x_emb: torch.Tensor) -> torch.Tensor:
    """
    对嵌入序列 x_emb (B, T, E) 逐步展开 RNN
    返回隐藏状态序列 H_seq (B, T, H)
    """
    B_, T_, E_ = x_emb.shape
    h = torch.zeros(B_, cell.hidden_size, device=x_emb.device)
    hs = []
    for t in range(T_):
        h = cell(x_emb[:, t, :], h)   # (B, H)
        hs.append(h.unsqueeze(1))      # (B, 1, H)
    H_seq = torch.cat(hs, dim=1)      # (B, T, H)
    return H_seq


# 验证 Unroll
cell_verify = MyRNNCell(E, H)
x_emb_v     = torch.randn(B, T, E)
H_seq_v     = rnn_unroll(cell_verify, x_emb_v)
assert H_seq_v.shape == (B, T, H), \
    f"H_seq shape 应为 ({B}, {T}, {H})，实际 {H_seq_v.shape}"
print(f"时间展开验证：x_emb={x_emb_v.shape} → H_seq={H_seq_v.shape}  ✓\n")

#5.完整模型结构
print("=" * 60)
print("【5】完整模型结构")
print("=" * 60)

class CharRNN(nn.Module):
    """字符级语言模型，核心 RNN Cell 完全手写"""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed   = nn.Embedding(vocab_size, embed_dim)
        self.rnn_cell = MyRNNCell(embed_dim, hidden_size)
        self.linear  = nn.Linear(hidden_size, vocab_size)  # 输出层，无 softmax

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
        """
        x   : (B, T)  整数序列
        h0  : (B, H) 初始隐藏状态，默认全零
        返回:
          logits : (B, T, V)
          h_last : (B, H) 最后时刻隐藏状态（用于文本生成）
        """
        B_, T_ = x.shape
        x_emb = self.embed(x)                              # (B, T, E)

        if h0 is None:
            h0 = torch.zeros(B_, self.hidden_size, device=x.device)

        H_seq = rnn_unroll(self.rnn_cell, x_emb)          # (B, T, H)
        assert H_seq.shape == (B_, T_, self.hidden_size)

        logits = self.linear(H_seq)                        # (B, T, V)
        h_last = H_seq[:, -1, :]                           # (B, H)
        return logits, h_last

    def forward_step(self, x_t: torch.Tensor, h: torch.Tensor):
        """
        单步推进（用于生成）
        x_t : (B,)  单字符整数
        h   : (B, H)
        返回 logit (B, V), h_new (B, H)
        """
        x_emb = self.embed(x_t)             # (B, E)
        h_new = self.rnn_cell(x_emb, h)     # (B, H)
        logit = self.linear(h_new)           # (B, V)
        return logit, h_new


model = CharRNN(vocab_size, E, H)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)
print(f"\n可训练参数总数: {total_params:,}\n")

#6.输出层与损失验证
print("=" * 60)
print("【6】输出层与损失验证")
print("=" * 60)

x_v, y_v = get_batch(batch_size=4, seq_len=T)
logits_v, _ = model(x_v)
print(f"logits shape: {logits_v.shape}  (B=4, T={T}, V={vocab_size})")

# reshape → CrossEntropyLoss
logits_flat = logits_v.reshape(-1, vocab_size)  # (B*T, V)
y_flat      = y_v.reshape(-1)                   # (B*T,)
loss_v = F.cross_entropy(logits_flat, y_flat)
print(f"初始 loss（随机权重，参考值 ≈ ln({vocab_size}) = {torch.log(torch.tensor(float(vocab_size))):.3f}）: {loss_v.item():.4f}\n")

#7.训练循环
print("=" * 60)
print("【7】训练循环")
print("=" * 60)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    for step in range(STEPS_PER_EPOCH):
        x_b, y_b = get_batch()               # (B, T)

        logits, _ = model(x_b)               # (B, T, V)

        # reshape 计算交叉熵
        loss = loss_criterion(
            logits.reshape(-1, vocab_size),   # (B*T, V)
            y_b.reshape(-1)                   # (B*T,)
        )

        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / STEPS_PER_EPOCH
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch [{epoch:3d}/{EPOCHS}]  avg_loss = {avg_loss:.4f}")

print("\n训练完成！\n")

#8.文本生成采样
print("=" * 60)
print("【8】文本生成采样")
print("=" * 60)

def sample(seed_text: str, gen_len: int = 200, temperature: float = 1.0) -> str:
    """
    从 seed_text 出发，逐字符生成 gen_len 个字符。
    策略：随机采样 multinomial(softmax(logits / temperature))
    """
    model.eval()
    generated = list(seed_text)

    with torch.no_grad():
        # —— 用 seed_text 预热隐藏状态 ——
        h = torch.zeros(1, H)
        for ch in seed_text:
            if ch not in stoi:
                continue   # 跳过未见字符
            x_t = torch.tensor([stoi[ch]], dtype=torch.long)  # (1,)
            _, h = model.forward_step(x_t, h)

        # —— 自回归生成 ——
        # 从最后一个 seed 字符的 logit 开始
        last_ch = seed_text[-1] if seed_text[-1] in stoi else vocab[0]
        x_t = torch.tensor([stoi[last_ch]], dtype=torch.long)

        for _ in range(gen_len):
            logit, h = model.forward_step(x_t, h)  # logit: (1, V)
            # temperature scaling + softmax → 采样
            probs = torch.softmax(logit[0] / temperature, dim=-1)  # (V,)
            next_id = torch.multinomial(probs, num_samples=1).item()
            next_ch = itos[next_id]
            generated.append(next_ch)
            x_t = torch.tensor([next_id], dtype=torch.long)

    return "".join(generated)


# 用文本开头作为种子
seed = TEXT[:10]
print(f"种子文本: {repr(seed)}\n")

print("── temperature = 1.0 ──")
out1 = sample(seed, gen_len=200, temperature=1.0)
print(out1)

print("\n── temperature = 0.5（更保守）──")
out2 = sample(seed, gen_len=200, temperature=0.5)
print(out2)

print("\n── temperature = 1.5（更随机）──")
out3 = sample(seed, gen_len=200, temperature=1.5)
print(out3)