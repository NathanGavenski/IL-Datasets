import torch
from torch import nn

from .Attention import Self_Attn2D


def normalize_imagenet(x: torch.Tensor, type: str = 'rgb') -> torch.Tensor:
    """Normalize tensor for imagenet networks"""
    mean = [0.485, 0.456, 0.406] if type == 'rgb' else 0.44531356896770125
    std = [0.229, 0.224, 0.225] if type == 'rgb' else 0.2692461874154524
    return (x - mean) / std


class CNN(nn.Module):
    """Default convolutional neural network."""

    def __init__(self, input, ):
        super().__init__()

        self.block0 = nn.Sequential(
            nn.Conv2d(in_channels=input[0], out_channels=32, kernel_size=8, stride=4),
            nn.LeakyReLU()
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            Self_Attn2D(64),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            Self_Attn2D(64),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU()
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = x.float()
        b, c, h, w = x.shape
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = x.view((b, -1))
        return x
