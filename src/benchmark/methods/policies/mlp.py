"""Module for mlp policies"""
import torch
from torch import nn

from .attention import SelfAttn1D


class MLP(nn.Module):
    """Default Multi Layer Perceptron"""

    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module = nn.LeakyReLU) -> None:
        """Initialize MLP class.

        Args:
            in_dim (int): input dimension.
            out_dim (int): output dimensions.
            activation (nn.Module): activation for the MLP. Defaults to LeakyReLU.
        """
        super().__init__()

        out = max(8, in_dim * 2)
        self.layers = nn.Sequential(
            nn.Linear(in_dim, out),
            activation(),

            nn.Linear(out, out),
            activation(),

            nn.Linear(out, out),
            activation(),

            nn.Linear(out, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward for default MLP

        Args:
            x (torch.Tensor): input (B, C).

        Returns:
            out (torch.Tensor): output (B, out_dim).
        """
        x = x.float()
        return self.layers(x)


class MlpWithAttention(nn.Module):
    """Default Multi Layer Perceptron with attention after first two layers."""

    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module = nn.LeakyReLU):
        """Initialize MlpWithAttention class.

        Args:
            in_dim (int): input dimensions.
            out_dim (int): output dimensions.
            activation (nn.Module): torch module for the activation layer.
        """
        super().__init__()

        out = max(8, in_dim * 2)
        self.layers = nn.Sequential(
            nn.Linear(in_dim, out),
            activation(),

            SelfAttn1D(out),
            nn.Linear(out, out),
            activation(),

            SelfAttn1D(out),
            nn.Linear(out, out),
            activation(),

            nn.Linear(out, out),
            activation(),

            nn.Linear(out, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for the method.

        Args:
            x (torch.Tensor): input for model.

        Returns:
            x (torch.Tensor): output from the model (B X out_dim).
        """
        x = x.float()
        return self.layers(x)


class MlpAttention(nn.Module):
    """Default Multi Layer Perceptron with attention as hidden layers."""

    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module = nn.LeakyReLU):
        """Initialize MlpAttention class.

        Args:
            in_dim (int): input dimensions.
            out_dim (int): output dimensions.
            activation (nn.Module): torch module for the activation layer.
        """
        super().__init__()

        out = max(8, in_dim * 2)
        self.layers = nn.Sequential(
            nn.Linear(in_dim, out),
            activation(),

            SelfAttn1D(out),
            activation(),

            SelfAttn1D(out),
            activation(),

            nn.Linear(out, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for the method.

        Args:
            x (torch.Tensor): input for model.

        Returns:
            x (torch.Tensor): output from the model (B X out_dim).
        """
        x = x.float()
        return self.layers(x)
