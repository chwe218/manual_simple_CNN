"""
dataset.py — CIFAR-10 数据加载

直接使用 torchvision 内置数据集，无需自定义 Dataset 类。

数据增强策略（仅训练集）:
    RandomHorizontalFlip  — 随机左右翻转，让模型不依赖位置
    RandomCrop(32, pad=4) — 先 padding 到 40×40 再随机裁回 32×32，
                            相当于随机平移，增强位置鲁棒性
    Normalize(0.5, 0.5)   — 将像素值从 [0,1] 映射到 [-1,1]，
                            加速收敛（零均值、单位方差更易优化）

测试集只做 ToTensor + Normalize，不做随机增强（保证评估可复现）。
"""

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ── 数据增强 ──────────────────────────────────────────────────────────────────
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ── 数据集 & DataLoader ───────────────────────────────────────────────────────
def get_dataloaders(batch_size: int = 64, data_root: str = './Cdataset'):
    """
    下载 CIFAR-10 并返回 (train_loader, test_loader)。

    Args:
        batch_size: 每个 mini-batch 的样本数，默认 64
        data_root:  数据集保存路径，默认 ./Cdataset
    """
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform_test
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# CIFAR-10 类别名，方便后续可视化用
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')