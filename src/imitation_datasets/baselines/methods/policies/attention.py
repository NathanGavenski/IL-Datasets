"""Module for attention layers."""
from typing import Tuple, Union

import torch
from torch import nn


class SelfAttn2D(nn.Module):
    """Self attention Layer"""

    def __init__(self, in_dim: int) -> None:
        """Initialize Self Attention with 2D convolutions.

        Args:
            in_dim (int): dimensions for the input.
        """
        super().__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False
    ) -> Union[Tuple[torch.Tensor], torch.Tensor]:
        """Forward function for the model.

        Args:
            x (torch.Tensor): input feature maps (B X C X W X H)

        Returns:
            out (torch.Tensor): self attention value + input feature
            attention (torch.Tensor): attention map (B X N X N, where N is Width * Height).
        """
        m_batchsize, C, width, height = x.size()

        # B X CX(N)
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height)
        proj_query = proj_query.permute(0, 2, 1)

        # B X C x (*W*H)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        # transpose check
        energy = torch.bmm(proj_query, proj_key)
        # BX (N) X (N)
        attention = self.softmax(energy)

        # B X C X N
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x

        if not return_attn:
            return out
        return out, attention


class SelfAttn1D(nn.Module):
    """Self attention Layer"""

    def __init__(self, in_dim: int, k: int = 8):
        """Initialize Self Attention with 1D Convolutions.

        Args:
            in_dim (int): dimensions for input.
            k (int): division factor for out_channels. Defaults to 8.
        """
        super().__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=in_dim // k,
            kernel_size=1,
        )
        self.key_conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=in_dim // k,
            kernel_size=1,
        )
        self.value_conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=1
        )
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False
    ) -> Union[Tuple[torch.Tensor], torch.Tensor]:
        """Forward function for the model.

        Args:
            x (torch.Tensor): input feature maps(B X C)
            return_attn (bool): whether to return the attention map Defaults to False.

        Returns:
            out (torch.Tensor): self-attention value + input feature.
            attention (torch.Tensor): attention map (B X N X N, where N is 1)
        """
        B, C = x.size()
        T = 1
        x = x.view(B, C, T)

        # B X C X (N)
        proj_query = self.query_conv(x).view(B, -1, T).permute(0, 2, 1)

        # B X C x (W*H)
        proj_key = self.key_conv(x).view(B, -1, T)
        energy = torch.bmm(proj_query, proj_key)

        # B X (N) X (N)
        attention = self.softmax(energy)

        # B X C X N
        proj_value = self.value_conv(x).view(B, -1, T)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, T)

        out = self.gamma * out + x
        out = out.squeeze(2)

        if not return_attn:
            return out
        return out, attention
