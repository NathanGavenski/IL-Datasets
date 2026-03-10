"""Module for mlp policies"""
from typing import List

import torch
from torch import nn

from .attention import SelfAttn1D


def create_layers(
    in_dim: int,
    out_dim: int,
    hidden_dim: List,
    activation: nn.Module = nn.LeakyReLU,
    self_attention: bool = False,
    only_attention: bool = False
) -> nn.Sequential:
    """Create layers for MLP.

    Args:
        in_dim (int): input dimension.
        out_dim (int): output dimensions.
        hidden_dim (List, optional): list of hidden dimensions. Defaults to None.
        activation (nn.Module, optional): activation for the MLP. Defaults to nn.LeakyReLU.
        self_attention (bool, optional): whether to include self-attention layers. Defaults to False.
        only_attention (bool, optional): whether to include only attention layers. Defaults to False.

    Returns:
        nn.Sequential: sequential of layers for MLP.
    """
    layers = nn.Sequential(
        nn.Linear(in_dim, hidden_dim[0]),
        activation()
    )
    for idx, dim in enumerate(hidden_dim):
        if self_attention:
            if not only_attention and idx <= len(hidden_dim) - 2:
                layers.add_module(f"attention_{idx}", SelfAttn1D(hidden_dim[idx-1]))
            elif only_attention:
                layers.add_module(f"attention_{idx}", SelfAttn1D(hidden_dim[idx-1]))
        if not only_attention:
            layers.add_module(f"linear_{idx}", nn.Linear(hidden_dim[idx-1], dim))
        layers.add_module(f"activation_{idx}", activation())
    layers.add_module("output", nn.Linear(hidden_dim[-1], out_dim))
    return layers


class MLP(nn.Module):
    """Default Multi Layer Perceptron"""

    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        hidden_dim: List = None,
        activation: nn.Module = nn.LeakyReLU
    ) -> None:
        """Initialize MLP class.

        Args:
            in_dim (int): input dimension.
            out_dim (int): output dimensions.
            activation (nn.Module): activation for the MLP. Defaults to LeakyReLU.
        """
        super().__init__()

        if hidden_dim is not None:
            self.layers = create_layers(in_dim, out_dim, hidden_dim, activation)
        else:
            out = max(32, in_dim * 2)
            self.layers = create_layers(in_dim, out_dim, [out, out], activation)

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

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: List = None,
        activation: nn.Module = nn.LeakyReLU
    ):
        """Initialize MlpWithAttention class.

        Args:
            in_dim (int): input dimensions.
            out_dim (int): output dimensions.
            activation (nn.Module): torch module for the activation layer.
        """
        super().__init__()

        if hidden_dim is not None:
            self.layers = create_layers(in_dim, out_dim, hidden_dim, activation, self_attention=True)
        else:
            out = max(8, in_dim * 2)
            self.layers = create_layers(in_dim, out_dim, [out, out, out], activation, self_attention=True)

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

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: List = None,
        activation: nn.Module = nn.LeakyReLU
    ):
        """Initialize MlpAttention class.

        Args:
            in_dim (int): input dimensions.
            out_dim (int): output dimensions.
            activation (nn.Module): torch module for the activation layer.
        """
        super().__init__()

        if hidden_dim is not None:
            self.layers = create_layers(
                in_dim, out_dim, hidden_dim, activation,
                self_attention=True, only_attention=True
            )
        else:
            out = max(8, in_dim * 2)
            self.layers = create_layers(
                in_dim, out_dim, [out, out], activation,
                self_attention=True, only_attention=True
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
