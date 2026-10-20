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


def get_loaders(batch_size=128):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    full = datasets.FashionMNIST(root="data", train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(root="data", train=False, download=True, transform=tf)
    g = torch.Generator().manual_seed(42)
    train, val = random_split(full, [55000, 5000], generator=g)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, generator=g),
        DataLoader(val, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size),
    )


def make_mlp():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )


def make_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 10),
    )


class PatchAttention(nn.Module):
    """28x28 image -> 16 patches of 7x7 -> Linear(49,64) -> 2 encoder layers
    (d_model=64, nhead=4, dim_feedforward=128) -> mean pool -> Linear(64,10)."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(49, 64)
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False
        )
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        B = x.shape[0]
        p = x.unfold(2, 7, 7).unfold(3, 7, 7)      # (B, 1, 4, 4, 7, 7)
        p = p.reshape(B, 16, 49)                   # 16 patches of 49 values
        z = self.encoder(self.proj(p))             # (B, 16, 64)
        return self.head(z.mean(dim=1))            # mean pool over patches


def make_attention():
    return PatchAttention()


MODELS = {"mlp": make_mlp, "cnn": make_cnn, "attention": make_attention}