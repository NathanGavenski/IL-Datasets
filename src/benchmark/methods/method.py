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
from gym import spaces as gym_spaces
import numpy as np
from imitation_datasets.utils import GymWrapper
from imitation_datasets.dataset import BaselineDataset
from .policies import MLP


Metrics = Dict[str, Any]


# TODO adapt for visual
class Method(ABC):
    """Base class for all methods."""

    def __init__(
        self,
        environment: Env,
        environment_parameters: Dict[str, Any],
        discrete_loss: nn.Module = nn.CrossEntropyLoss,
        continuous_loss: nn.Module = nn.MSELoss,
        optimizer_fn: optim.Optimizer = optim.Adam,
    ) -> None:
        """Initialize base class."""
        super().__init__()
        self.environment = environment
        self.discrete = isinstance(environment.action_space, spaces.Discrete)
        self.discrete |= isinstance(environment.action_space, gym_spaces.Discrete)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        observation_size = environment.observation_space.shape[0]

        if self.discrete:
            action_size = environment.action_space.n
            self.loss_fn = discrete_loss()
        else:
            action_size = environment.action_space.shape[0]
            self.loss_fn = continuous_loss()

        self.policy = MLP(observation_size, action_size)

        self.optimizer_fn = optimizer_fn(
            self.policy.parameters(),
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
        self.policy.eval()

        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)

            if len(obs.shape) == 1:
                obs = obs[None]

        obs = obs.to(self.device)

        with torch.no_grad():
            actions = self.forward(obs)
            actions = actions[0]

            if self.discrete:
                return torch.argmax(actions).cpu().numpy()
            return actions.cpu().numpy()

    def save(self, path: str = None) -> None:
        """Save all model weights.

        Args:
            path (str): where to save the models. Defaults to None.
        """
        raise NotImplementedError()

    def load(self, path: str = None) -> None:
        """Load all model weights.

        Args:
            path (str): where to look for the model's weights. Defaults to None.
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

    def _train(self, dataset: DataLoader[BaselineDataset]) -> Metrics:
        """Train loop.

        Args:
            dataset (DataLoader): train data.
        """
        raise NotImplementedError()

    def _eval(self, dataset: DataLoader[BaselineDataset]) -> Metrics:
        """Evaluation loop.

        Args:
            dataset (DataLoader): data to eval.
        """
        raise NotImplementedError()

    def _enjoy(self, render: bool = False) -> Metrics:
        """Function for evaluation of the policy in the environment

        Args:
            render (bool): Whether it should render. Defaults to False.

        Returns:
            Metrics:
                aer (Number): average reward for 10 episodes.
        """
        environment = GymWrapper(self.environment)
        average_reward = []
        for _ in range(10):
            done = False
            obs = environment.reset()
            accumulated_reward = 0
            while not done:
                if render:
                    environment.render()
                action = self.predict(obs)
                obs, reward, done, *_ = environment.step(action)
                accumulated_reward += reward
            average_reward.append(accumulated_reward)
        return {"aer": np.mean(average_reward)}
