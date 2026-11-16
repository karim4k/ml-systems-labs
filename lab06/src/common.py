import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders(batch_size=128, num_workers=0, seed=42):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    full = datasets.FashionMNIST(root="data", train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(root="data", train=False, download=True, transform=tf)
    g = torch.Generator().manual_seed(seed)
    train, val = random_split(full, [55000, 5000], generator=g)
    kw = dict(num_workers=num_workers, persistent_workers=num_workers > 0)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, generator=g, **kw),
        DataLoader(val, batch_size=512, **kw),
        DataLoader(test, batch_size=512, **kw),
    )


def make_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 10),
    )