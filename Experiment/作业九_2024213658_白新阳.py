import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']

# 0. 超参数与全局配置
torch.manual_seed(42)
np.random.seed(42)

VOCAB   = 20      # token 词表大小：0–19
L       = 16      # 序列长度
D_MODEL = 64      # embedding 维度
D_K     = 32      # Q/K 投影维度
D_V     = 32      # V 投影维度
BATCH   = 256     # 训练 batch size
N_TRAIN = 8000    # 训练样本数
N_TEST  = 1000    # 测试样本数
EPOCHS  = 30
LR      = 1e-3

# 1. 核心函数：缩放点积注意力
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力

    Args:
        Q    : (B, Tq, d_k)
        K    : (B, Tk, d_k)
        V    : (B, Tk, d_v)
        mask : (B, Tq, Tk)，True/1 表示"保留"，False/0 表示"屏蔽"
               若为 None 则不做任何屏蔽

    Returns:
        out  : (B, Tq, d_v)   注意力加权后的输出
        attn : (B, Tq, Tk)    softmax 后的注意力权重
    """
    d_k = Q.size(-1)

    # Step 1: 计算相似度分数  (B, Tq, Tk)
    scores = torch.bmm(Q, K.transpose(1, 2)) / (d_k ** 0.5)

    # Step 2: 应用 mask（masked 位置加极大负数，softmax 后接近 0）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # Step 3: softmax 在最后一维 (Tk) 上做
    attn = F.softmax(scores, dim=-1)   # (B, Tq, Tk)

    # Step 4: 加权求和
    out = torch.bmm(attn, V)           # (B, Tq, d_v)

    return out, attn


# ---- 形状验证 ----
def verify_attention_shapes():
    print("=" * 60)
    print("【1】验证 scaled_dot_product_attention 张量形状")
    print("=" * 60)
    B, Tq, Tk, d_k, d_v = 4, 3, 8, 16, 32
    Q = torch.randn(B, Tq, d_k)
    K = torch.randn(B, Tk, d_k)
    V = torch.randn(B, Tk, d_v)

    out, attn = scaled_dot_product_attention(Q, K, V)

    print(f"  Q shape   : {tuple(Q.shape)}")
    print(f"  K shape   : {tuple(K.shape)}")
    print(f"  V shape   : {tuple(V.shape)}")
    print(f"  out shape : {tuple(out.shape)}   期望: ({B}, {Tq}, {d_v})")
    print(f"  attn shape: {tuple(attn.shape)}  期望: ({B}, {Tq}, {Tk})")
    print(f"  attn.sum(dim=-1):\n{attn.sum(dim=-1)}")   # 每行应≈1
    assert out.shape  == (B, Tq, d_v), "out shape 错误！"
    assert attn.shape == (B, Tq, Tk),  "attn shape 错误！"
    assert torch.allclose(attn.sum(dim=-1), torch.ones(B, Tq), atol=1e-5), \
        "attn 行和不为 1！"
    print("  ✓ 形状验证通过\n")

# 2. 注意力层（单头版本）
class AttentionLayer(nn.Module):
    """
    单头注意力层
    输入 X: (B, L, d_model)
    输出 H: (B, L, d_v)
    """
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_k, bias=False)
        self.Wk = nn.Linear(d_model, d_k, bias=False)
        self.Wv = nn.Linear(d_model, d_v, bias=False)

    def forward(self, X, mask=None):
        Q = self.Wq(X)   # (B, L, d_k)
        K = self.Wk(X)   # (B, L, d_k)
        V = self.Wv(X)   # (B, L, d_v)
        H, attn = scaled_dot_product_attention(Q, K, V, mask)
        return H, attn   # (B, L, d_v), (B, L, L)
    
# 3. 合成数据生成
def generate_dataset(n_samples):
    """
    生成指针检索数据集

    Returns:
        x : (n_samples, L)  token 序列，值域 [0, VOCAB)
        p : (n_samples,)    指针位置，值域 [0, L)
        y : (n_samples,)    目标 token = x[i, p[i]]
    """
    x = torch.randint(0, VOCAB, (n_samples, L))
    p = torch.randint(0, L,    (n_samples,))
    y = x[torch.arange(n_samples), p]
    return x, p, y