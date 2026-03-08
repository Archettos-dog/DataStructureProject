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