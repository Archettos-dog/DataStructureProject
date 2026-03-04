import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

#0.可复现性
torch.manual_seed(42)

#1.载入数据&划分&标准化
print("=" * 50)
print("Step 1 — 数据载入与预处理")
print("=" * 50)

X, y = load_digits(return_X_y=True)          # X: (1797, 64), y: (1797,)

# 先划出 test (20%)，再把剩余部分划出 val (12.5% → 整体约 10%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.125, random_state=42, stratify=y_train_val
)

# StandardScaler（用 train 集 fit）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

print(f"X_train: {X_train.shape},  y_train: {y_train.shape}")
print(f"X_val  : {X_val.shape},    y_val  : {y_val.shape}")
print(f"X_test : {X_test.shape},   y_test : {y_test.shape}")

#2.转Tensor&构建DataLoader
print("\n" + "=" * 50)
print("Step 2 — Tensor 转换 & DataLoader")
print("=" * 50)

# X → float32，y → long
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_val_t   = torch.tensor(y_val,   dtype=torch.long)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

# 演示 Tensor 基本操作
print(f"\n[Tensor 基本操作演示]")
print(f"  X_train_t.shape  : {X_train_t.shape}")
print(f"  X_train_t.dtype  : {X_train_t.dtype}")
print(f"  X_train_t.mean() : {X_train_t.mean():.4f}")
print(f"  X_train_t.std()  : {X_train_t.std():.4f}")
print(f"  前3行前5列:\n{X_train_t[:3, :5]}")

# 演示 Autograd
print(f"\n[Autograd 演示]")
w = torch.tensor([2.0, 3.0], requires_grad=True)
loss_demo = (w ** 2).sum()          # scalar
loss_demo.backward()
print(f"  w         : {w.data}")
print(f"  w.grad    : {w.grad}")    # dl/dw = 2w → [4, 6]
w.grad.zero_()                      # 清零梯度
print(f"  zero_grad 后 w.grad: {w.grad}")

# TensorDataset & DataLoader
train_ds = TensorDataset(X_train_t, y_train_t)
val_ds   = TensorDataset(X_val_t,   y_val_t)
test_ds  = TensorDataset(X_test_t,  y_test_t)

train_loader = DataLoader(train_ds, batch_size=64,  shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)

print(f"\n  train_loader: {len(train_loader)} batches  "
      f"(batch_size=64, shuffle=True)")
print(f"  val_loader  : {len(val_loader)} batches  "
      f"(batch_size=256, shuffle=False)")

#3.定义MLP模型
print("\n" + "=" * 50)
print("Step 3 — 定义 MLP (64→128→ReLU→10)")
print("=" * 50)

class MLP(nn.Module):
    def __init__(self, in_dim: int = 64, hidden_dim: int = 128, out_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

model     = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"可训练参数量: {total_params:,}")

#4.训练循环
print("\n" + "=" * 50)
print("Step 4 — 训练")
print("=" * 50)

NUM_EPOCHS = 30
train_losses = []

for epoch in range(1, NUM_EPOCHS + 1):
    # ── Train ──
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()           # 清零梯度
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()                 # 反向传播
        optimizer.step()               # 参数更新
        epoch_loss += loss.item() * len(X_batch)

    avg_train_loss = epoch_loss / len(train_ds)
    train_losses.append(avg_train_loss)

    # ── Validation ──
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            logits = model(X_batch)
            preds  = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total   += len(y_batch)
    val_acc = correct / total

    print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}]  "
          f"Train Loss: {avg_train_loss:.4f}  |  "
          f"Val Acc: {val_acc*100:.2f}%")

#5.测试评估
print("\n" + "=" * 50)
print("Step 5 — 测试集评估")
print("=" * 50)

def evaluate(loader, mdl):
    mdl.eval()
    correct = total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds   = mdl(X_batch).argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total   += len(y_batch)
    return correct / total

test_acc = evaluate(test_loader, model)
print(f"Test Accuracy: {test_acc*100:.2f}%")

