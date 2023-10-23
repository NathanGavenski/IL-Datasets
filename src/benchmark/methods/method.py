"""Module for base class for all methods."""
from abc import ABC
from typing import List, Dict, Union, Any

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from numbers import Number

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from gymnasium import Env, spaces
import numpy as np


Metrics = Dict[str, Any]


# TODO Define type dataset.
class Method(ABC):
    """Base class for all methods."""

    def __init__(
        self,
        environment: Env,
        model_parameters: nn.Parameter,
        environment_parameters: Dict[str, Any],
        discrete_loss: nn.Module = nn.CrossEntropyLoss,
        continuous_loss: nn.Module = nn.MSELoss,
        optimizer_fn: optim.Optimizer = optim.Adam,
    ) -> None:
        """Initialize base class."""
        super().__init__()
        self.loss_fn: nn.Module = None
        if isinstance(environment.action_space, spaces.Discrete):
            self.loss_fn = discrete_loss()
        else:
            self.loss_fn = continuous_loss()

        self.optimizer_fn = optimizer_fn(
            model_parameters,
            **environment_parameters
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for the method.

        Args:
            x (torch.Tensor): input.

        Returns:
            x (torch.Tensor): logits output.
        """
        raise NotImplementedError()

    def predict(self, obs: Union[np.ndarray, torch.Tensor]) -> Union[List[Number], Number]:
        """Predict method.

        Args:
            obs (Union[np.ndarray, torch.Tensor]): input observation.

        Returns:
            action (Union[List[Number], Number): predicted action.
        """
        raise NotImplementedError()

    # pylint: disable=W0221
    def train(
        self,
        n_epochs: int,
        train_dataset: DataLoader,
        eval_dataset: DataLoader = None
    ) -> Self:
        """Train process.

        Args:
            n_epochs (int): amount of epoch to run.
            train_dataset (DataLoader): data to train.
            eval_dataset (DataLoader): data to eval. Defaults to None.

        Returns:
            method (Self): trained method.
        """
        raise NotImplementedError()

    def _train(self, dataset: DataLoader) -> Metrics:
        """Train loop.

        Args:
            dataset (DataLoader): train data.
        """
        raise NotImplementedError()

    def _eval(self, dataset: DataLoader) -> Metrics:
        """Evaluation loop.

        Args:
            dataset (DataLoader): data to eval.
        """
        raise NotImplementedError()
