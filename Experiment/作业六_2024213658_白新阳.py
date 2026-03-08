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
