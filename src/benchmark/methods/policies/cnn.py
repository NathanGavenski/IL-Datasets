"""Module for CNN based policies."""
from typing import Tuple

import torch
from torch import nn
from torchvision import models, transforms

from .attention import SelfAttn2D


class Empty(nn.Module):
    """Empty module for removing classifying layers from torchvision models."""

    def __init__(self):
        """Empty layer"""
        super().__init__()

    def forward(self, x):
        """It just returns X without doing anything.

        Args:
            x (torch.Tensor): input.

        Returns:
            x (torch.Tensor): return the exact same input.
        """
        return x


def convert_to_bw(model: nn.Module) -> nn.Module:
    """Convert first layer of a model to accept bw input.

    Args:
        model (nn.Module): model to convert layer.

    Returns:
        model (nn.Module): model with new layer.
    """
    conv1_weights = model.conv1.weight
    conv1_weights = model.conv1.weight.sum(dim=1, keepdim=True)
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
        bias=False
    )
    return model


def normalize_imagenet(x: torch.Tensor, mode: str = 'rgb') -> torch.Tensor:
    """Normalize tensor for imagenet networks

    Args:
        x (torch.Tensor): input tensor (B, C, W, H)
        mode (str): Mode to normalize (bw or rgb). Defaults to rgb.

    Raises:
        ValueError: if x shape and mode do not match.

    Returns:
        normalized (torch.Tensor): normalized input tensor.
    """
    if mode == "rgb" and x.size(1) != 3:
        raise ValueError("Input and mode do not match in shape.")

    if mode == "bw" and x.size(1) != 1:
        raise ValueError("Input and mode do not match in shape.")

    mean = [0.485, 0.456, 0.406] if mode == 'rgb' else 0.44531356896770125
    std = [0.229, 0.224, 0.225] if mode == 'rgb' else 0.2692461874154524
    return transforms.Normalize(mean, std)(x)


class CNN(nn.Module):
    """DQN convolutional neural network."""

    def __init__(
        self,
        input_shape: Tuple[int] = (84, 84, 4),
        activation: nn.Module = nn.LeakyReLU
    ):
        """Vanilla DQN CNN.

        Args:
            input_shape (Tuple[int]): input shape (C, W, H).
            activation (nn.Module): activation layers. Defaults to nn.LeakyReLU.
        """
        super().__init__()

        self.block0 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[-1], out_channels=32, kernel_size=8, stride=4),
            activation()
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64, affine=True),
            activation()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64, affine=True),
            activation()
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


class Resnet(nn.Module):
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
        elif input_shape[0] == 1:
            self.model = convert_to_bw(models.resnet18(pretrained=pretrained))
        else:
            self.model = models.resnet18(pretrained=pretrained)
        self.model.fc = Empty()

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
        return self.model(x)


class ResnetWithAttention(nn.Module):
    """Resnet with two self-attention layers."""

    def __init__(
        self,
        input_shape: Tuple[int],
        normalize: bool = False,
        pretrained: bool = False
    ):
        """Renset 18 with two self-attention layers after the first and second ResBlock.

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
        elif input_shape[0] == 1:
            self.model = convert_to_bw(models.resnet18(pretrained=pretrained))
        else:
            self.model = models.resnet18(pretrained=pretrained)
        self.model.fc = Empty()

        self.mode = "bw" if input_shape[0] else "rgb"

        name = str(len(self.model.layer1))
        self.model.layer1.add_module(name, SelfAttn2D(64))

        name = str(len(self.model.layer2))
        self.model.layer2.add_module(name, SelfAttn2D(128))

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
        return self.model(x)
