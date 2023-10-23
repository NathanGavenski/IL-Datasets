"""Module for Behavioural Cloning"""
import os
from typing import List, Union
from numbers import Number

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import torch
from torch.utils.data import DataLoader
from gymnasium import Env, spaces
from tensorboard_wrapper.tensorboard import Tensorboard

from .policies.mlp import MLP
from .method import Method, Metrics


# TODO adapt for visual
class BC(Method):
    """Behavioural Clonning method based on (Pomerleau, 1988)"""

    def __init__(self, environment: Env) -> None:
        """Initialize BC method."""
        self.environment = environment
        self.discrete = isinstance(environment.action_space, spaces.Discrete)
        self.observation_size = environment.observation_space.shape[0]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.action_size = None
        if self.discrete:
            self.action_size = environment.action_space.n
        else:
            self.action_size = environment.action_space.shape[0]

        self.policy = MLP(self.observation_size, self.action_size)
        super().__init__(
            self.environment,
            self.policy.parameters(),
            {"lr": 1e-3}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for the method.

        Args:
            x (torch.Tensor): input.

        Returns:
            x (torch.Tensor): logits output.
        """
        return self.policy(x)

    def predict(self, obs: Union[np.ndarray, torch.Tensor]) -> Union[List[Number], Number]:
        """Predict method.

        Args:
            obs (Union[np.ndarray, torch.Tensor]): input observation.

        Returns:
            action (Union[List[Number], Number): predicted action.
        """
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)

            if len(obs.shape) == 1:
                obs = obs[None]

        with torch.no_grad:
            actions = self.forward(obs)
            actions = actions[0]

            if self.discrete:
                return torch.argmax(actions).numpy()
            return actions.numpy()

    def train(
        self,
        n_epochs: int,
        train_dataset: DataLoader,
        eval_dataset: DataLoader = None,

    ) -> Self:
        """Train process.

        Args:
            n_epochs (int): amount of epoch to run.
            train_dataset (DataLoader): data to train.
            eval_dataset (DataLoader): data to eval. Defaults to None.

        Returns:
            method (Self): trained method.
        """
        folder = "../benchmark_results/bc/"
        if not os.path.exists(folder):
            os.makedirs(f"{folder}/")
        board = Tensorboard(path=folder)

        for _ in range(n_epochs):
            train_metrics = self._train(train_dataset)
            board.add_scalars("Train", epoch="train", **train_metrics)

            eval_metrics = self._eval(eval_dataset)
            board.add_scalars("Eval", epoch="eval", **eval_metrics)
            board.step()

        return self

    def _train(self, dataset: DataLoader) -> Metrics:
        """Train loop.

        Args:
            dataset (DataLoader): train data.
        """
        accumulated_loss = []
        accumulated_accuracy = []

        for batch in dataset:
            state, action, _ = batch
            state = state.to(self.device)
            action = action.to(self.device)

            self.optimizer_fn.zero_grad()
            predictions = self.forward(state)

            predictions_argmax = torch.argmax(predictions, 1)
            loss = self.loss_fn(predictions, action.long())
            loss.backward()
            accumulated_loss.append(loss.item())

            accuracy = ((predictions_argmax == action).sum().item() / action.size(0)) * 100
            accumulated_accuracy.append(accuracy)

        return {"loss": np.mean(accumulated_loss), "accuracy": np.mean(accumulated_accuracy)}

    def _eval(self, dataset: DataLoader) -> Metrics:
        """Evaluation loop.

        Args:
            dataset (DataLoader): data to eval.
        """
        accumulated_accuracy = []

        for batch in dataset:
            state, action, _ = batch
            state = state.to(self.device)

            with torch.no_grad:
                predictions = self.policy(state)
            predictions_argmax = torch.argmax(predictions, 1)
            accuracy = ((predictions_argmax == action).sum().item() / action.size(0)) * 100
            accumulated_accuracy.append(accuracy)

        return {"accuracy": np.mean(accumulated_accuracy)}
