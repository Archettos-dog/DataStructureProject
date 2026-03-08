import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#1.模型定义
class LeNet5(nn.Module):
    """
    LeNet-5 适配 FashionMNIST（28×28 灰度图，10 类）

    各层输出维度（batch=N）：
      输入          : (N,  1, 28, 28)
      Conv1(p=2)    : (N,  6, 28, 28)   # (28+2*2-5)/1+1 = 28
      Pool1         : (N,  6, 14, 14)
      Conv2(p=0)    : (N, 16, 10, 10)   # (14-5)/1+1    = 10
      Pool2         : (N, 16,  5,  5)
      Flatten       : (N, 400)           # 16×5×5 = 400
      FC1           : (N, 120)
      FC2           : (N,  84)
      FC3(output)   : (N,  10)
    """

    def __init__(self):
        super(LeNet5, self).__init__()

        # ── Layer 1 ──
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6,
                      kernel_size=5, stride=1, padding=2),  # (N,6,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)            # (N,6,14,14)
        )

        # ── Layer 2 ──
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16,
                      kernel_size=5, stride=1, padding=0),   # (N,16,10,10)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)            # (N,16,5,5)
        )

        # ── Layer 3：Flatten ──
        self.flatten = nn.Flatten()                          # (N,400)

        # ── Layer 4：全连接层 ──
        self.fc = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)                                # 10 类输出
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
#2，维度自检函数
def check_shape(model: LeNet5):
    """
    输入随机张量 (1,1,28,28)，逐层打印并断言输出 shape。
    """
    print("=" * 50)
    print("  维度自检 check_shape()")
    print("=" * 50)

    x = torch.randn(1, 1, 28, 28)
    print(f"  输入          : {tuple(x.shape)}")
    assert x.shape == (1, 1, 28, 28), "输入维度错误"

    x = model.layer1(x)
    print(f"  Layer1 输出   : {tuple(x.shape)}")
    assert x.shape == (1, 6, 14, 14), f"Layer1 维度异常: {x.shape}"

    x = model.layer2(x)
    print(f"  Layer2 输出   : {tuple(x.shape)}")
    assert x.shape == (1, 16, 5, 5), f"Layer2 维度异常: {x.shape}"

    x = model.flatten(x)
    print(f"  Flatten 输出  : {tuple(x.shape)}")
    assert x.shape == (1, 400), f"Flatten 维度异常: {x.shape}"

    x = model.fc(x)
    print(f"  FC 输出       : {tuple(x.shape)}")
    assert x.shape == (1, 10), f"FC 维度异常: {x.shape}"

    print("  ✓ 所有维度断言通过！")
    print("=" * 50)

#3.数据加载
def get_dataloaders(batch_size: int = 64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader

#4.训练、测试函数
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    accuracy = correct / len(loader.dataset)
    return accuracy

#5.主训练pipeline
def main():
    # ── 超参数 ──
    EPOCHS     = 10
    BATCH_SIZE = 64
    LR         = 0.001

    # ── 设备检测 ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  使用设备: {device.upper()}\n")

    # ── 数据集 ──
    train_loader, test_loader = get_dataloaders(BATCH_SIZE)
    print(f"  训练集大小: {len(train_loader.dataset)}")
    print(f"  测试集大小: {len(test_loader.dataset)}\n")

    # ── 模型 ──
    model = LeNet5().to(device)

    # ── 维度自检（CPU 即可）──
    check_shape(LeNet5())   # 用 CPU 副本检查，不影响主模型
    print()

    # ── 损失函数 & 优化器 ──
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ── 训练循环 ──
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Test Acc':>10}")
    print("-" * 34)
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_acc   = evaluate(model, test_loader, device)
        print(f"{epoch:>6}  {train_loss:>12.4f}  {test_acc:>9.2%}")

    print("\n  训练完成！")

    # ── 保存模型权重（可选）──
    torch.save(model.state_dict(), "lenet5_fashionmnist.pth")
    print("  模型权重已保存至 lenet5_fashionmnist.pth")


if __name__ == "__main__":
    main()