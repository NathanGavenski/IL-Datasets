"""Module for CNN based policies."""
from typing import Tuple

import torch
from torch import nn
from torchvision import models


def convert_to_bw(model: nn.Module) -> nn.Module:
    """Convert first layer of a model to accept bw input.

    Args:
        model (nn.Module): model to convert layer.

    Returns:
        model (nn.Module): model with new layer.
    """
    conv1_weights = model.conv1.weight
    conv1_weights = model.conve1.weight(dim=1, keepdim=True)
    model.conv1.weight = nn.Parameter(conv1_weights)
    model.conv1.in_channels = 1
    return model


def convert_to_n_channels(model: nn.Module, n_channels: int) -> nn.Module:
    """Convert first layer of a model to accept new input.

    Args:
        model (nn.Module): model to convert layer.
        n_channels (int): size of the new input.

    Returns:
        model (nn.Module): model with new layer.
    """
    model.conv1 = torch.nn.Conv2d(
        n_channels,
        64,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
    )
    return model


def normalize_imagenet(x: torch.Tensor, mode: str = 'rgb') -> torch.Tensor:
    """Normalize tensor for imagenet networks"""
    mean = [0.485, 0.456, 0.406] if mode == 'rgb' else 0.44531356896770125
    std = [0.229, 0.224, 0.225] if mode == 'rgb' else 0.2692461874154524
    return (x - mean) / std


class CNN(nn.Module):
    """DQN convolutional neural network."""

    def __init__(
        self,
        input_shape: Tuple[int] = (4, 84, 84),
        activation: nn.Module = nn.LeakyReLU
    ):
        """Vanilla DQN CNN.

        Args:
            input_shape (Tuple[int]): input shape (C, W, H).
            activation (nn.Module): activation layers. Defaults to nn.LeakyReLU.
        """
        super().__init__()
        self.activation = activation

        self.block0 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),
            self.activation()
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64, affine=True),
            self.activation()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64, affine=True),
            self.activation()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward of the model.

        Args:
            x (torch.Tensor): input for the model.

        Returs:
            x (torch.Tensor): output of the model.
        """
        x = x.float()
        B, *_ = x.shape
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = x.view((B, -1))
        return x


class Restnet(nn.Module):
    """Resnet convolutional neural network."""

    def __init__(
        self,
        input_shape: Tuple[int],
        normalize: bool = False,
        pretrained: bool = False
    ):
        """Vanilla Resnet 18.

        Args:
            input_shape (Tuple[int]): input shape (C, W, H).
            normalize (bool): whether to normalize the input. Defaults to False.
            pretrained (bool): whether to use imagenet pretrained weights. Defaults to False.

        Raises:
            ValueError: if input_shape is not 3 dimensional.
        """
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError("Input should be a 3 dimensional tuple (C, W, H)")

        self.normalize = normalize

        self.model = None
        if input_shape[0] not in [1, 3]:
            self.model = convert_to_n_channels(
                models.resnet18(pretrained=False),
                input_shape[0]
            )
        if input_shape[0] == 1:
            self.model = convert_to_bw(models.resnet18(pretrained=pretrained))
        else:
            self.model = models.resnet18(pretrained=pretrained)
        self.mode = "bw" if input_shape[0] else "rgb"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward of the model.

        Args:
            x (torch.Tensor): input for the model.

        Returs:
            x (torch.Tensor): output of the model.
        """
        x = x.float()
        if self.normalize:
            x = normalize_imagenet(x, self.mode)

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x
