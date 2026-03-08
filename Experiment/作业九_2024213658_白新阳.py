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

