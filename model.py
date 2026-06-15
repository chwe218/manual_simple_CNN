"""
model.py — SimpleCNN 模型定义

网络结构（手写，不用预训练权重）：
    Input: 3×32×32 (CIFAR-10 RGB)

    Conv1: Conv2d(3→16, k=3) → ReLU → MaxPool(2×2)
           输出尺寸: (32-2)/2 = 15 → 16×15×15

    Conv2: Conv2d(16→32, k=3) → ReLU → MaxPool(2×2)
           输出尺寸: (15-2)/2 = 6 → 32×6×6

    Flatten → FC(32*6*6=1152 → 10)

为什么这样设计:
    - 每个 Conv 后接 MaxPool，逐步缩小 spatial size，增大感受野
    - 通道数 3→16→32，逐层提取更抽象的特征
    - 最后一层直接输出 10 个 logit（CrossEntropyLoss 内部会做 softmax）
"""

import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fl = nn.Flatten()
        self.fc = nn.Linear(32 * 6 * 6, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.fl(x)
        x = self.fc(x)
        return x