import torch
import torch.nn as nn

from .attention import Self_Attn1D


class MLP(nn.Module):
    """Default Multi Layer Perceptron"""

    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module = nn.LeakyReLU) -> None:
        """Initialize MLP class.

        Args:
            in_dim (int): input dimension.
            out_dim (int): output dimensions.
            activation (nn.Module): activation for the MLP. Defaults to LeakyReLU.
        """
        super(MLP, self).__init__()

        out = max(8, in_dim * 2)
        self.input = nn.Linear(in_dim, out)
        self.fc = nn.Linear(out, out)
        self.fc2 = nn.Linear(out, out)
        self.output = nn.Linear(out, out_dim)

        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward for default MLP

        Args:
            x (torch.Tensor): input (B, C).

        Returns:
            out (torch.Tensor): output (B, out_dim).
        """
        x = x.float()
        x = self.activation(self.input(x))
        x = self.activation(self.fc(x))
        x = self.activation(self.fc2(x))
        x = self.output(x)
        return x


class MlpWithAttention(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module = nn.LeakyReLU):
        """Initialize MlpWithAttention class.

        Args:
            in_dim (int): 
        """
        super(MlpWithAttention, self).__init__()

        out = max(8, in_dim * 2)
        self.input = nn.Linear(in_dim, out)
        self.output = nn.Linear(out, out_dim)

        self.fc = nn.Linear(out, out)
        self.fc2 = nn.Linear(out, out)
        self.fc3 = nn.Linear(out, out)
        self.attention = Self_Attn1D(out)
        self.attention2 = Self_Attn1D(out)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = x.float()
        x = self.relu(self.input(x))
        x = self.attention(x)
        x = self.relu(self.fc(x))
        x = self.attention2(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.output(x))
        return x


class MlpAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MlpAttention, self).__init__()

        out = max(8, in_dim * 2)
        self.input = nn.Linear(in_dim, out)
        self.output = nn.Linear(out, out_dim)
        self.attention = Self_Attn1D(out)
        self.attention2 = Self_Attn1D(out)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = x.float()
        x = self.input(x)
        x = self.attention(x)
        x = self.attention2(x)
        x = self.relu(self.output(x))
        return x
