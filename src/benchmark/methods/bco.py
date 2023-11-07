"""Module for Behavioural Cloning from Observation"""
import os
from typing import List, Union, Dict
from numbers import Number

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from gymnasium import Env, spaces
from tensorboard_wrapper.tensorboard import Tensorboard

from .policies.mlp import MLP
from .method import Metrics, Method


class BCO(Method):
    """Behavioural Cloning from Observation method based on (Torabi et. al., 2018)"""

    __version__ = "1.0.0"
    __author__ = "Torabi et. al."
    __method_name__ = "Behavioural Cloning from Observation"


    def __init__(self, environment: Env) -> None:
        """Initialize BCO method."""
        self.environment = environment
        self.discrete = isinstance(environment.action_space, spaces.Discrete)
        self.observation_size = environment.observation_space.shape[0]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.action_size = None
        if self.discrete:
            self.action_size = environment.action_space.n
        else:
            self.action_size = environment.action_space.shape[0]

        self.idm = MLP(self.observation_size * 2, self.action_size)
        self.idm_optimizer = optim.Adam(self.idm.parameters(), lr=1e-3)
        self.idm_loss = nn.CrossEntropyLoss() if self.discrete else nn.MSELoss()

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

    def save(self, path: str = None) -> None:
        """Save all models weights.

        Args:
            path (str): where to save the models. Defaults to None.
        """
        path = self.save_path if path is None else path
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.policy.state_dict(), f"{path}/best_model.ckpt")
        torch.save(self.idm.state_dict(), f"{path}/idm.ckpt")

    def load(self, path: str = None) -> Self:
        """Load all model weights.

        Args:
            path (str): where to look for the model's weights. Defaults to None.

        Raises:
            ValueError: if the path does not exist.
        """
        path = self.save_path if path is None else path

        if not os.path.exists(path):
            raise ValueError("Path does not exists.")

        self.policy.load_state_dict(
            torch.load(
                f"{path}best_model.ckpt",
                map_location=torch.device(self.device)
            )
        )
        self.idm.load_state_dict(
            torch.load(
                f"{path}/idm.ckpt",
                map_location=torch.device(self.device)
            )
        )

    def train(
        self,
        n_epochs: int,
        train_dataset: Dict[str, DataLoader],
        eval_dataset: Dict[str, DataLoader] = None,
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
            train_metrics = self._train(**train_dataset)
            board.add_scalars("Train", epoch="train", **train_metrics)

            eval_metrics = self._eval(eval_dataset)
            board.add_scalars("Eval", epoch="eval", **eval_metrics)

            # Enjoy and append to the dataset (time to create the dataset)
            board.step()

        return self

    def _train(self, idm_dataset: DataLoader, expert_dataset: DataLoader) -> Metrics:
        """Train loop.

        Args:
            dataset (DataLoader): train data.
        """
        idm_accumulated_loss = []
        idm_accumulated_accuracy = []

        accumulated_loss = []
        accumulated_accuracy = []

        for batch in idm_dataset:
            state, action, next_state = batch
            state = state.to(self.device)
            action = action.to(self.device)
            next_state = next_state.to(self.device)

            self.idm_optimizer.zero_grad()
            predictions = self.idm(torch.cat((state, next_state), dim=1))

            loss = self.idm_loss(predictions, action)
            loss.backward()
            idm_accumulated_loss.append(loss.item())
            self.idm_optimizer.step()

            accuracy: Number = None
            if self.discrete:
                predictions_argmax = torch.argmax(predictions, 1)
                accuracy = ((predictions_argmax == action).sum().item() / action.size(0)) * 100
            else:
                accuracy = (action - predictions).pow(2).sum(1).sqrt().mean().item()
            idm_accumulated_accuracy.append(accuracy)

        for batch in expert_dataset:
            state, _, next_state = batch
            state = state.to(self.device)
            next_state = next_state.to(self.device)

            with torch.no_grad:
                if self.discrete:
                    action = torch.argmax(self.idm(torch.cat((state, next_state), dim=1)))
                else:
                    action = self.idm(torch.cat((state, next_state), dim=1))

            self.optimizer_fn.zero_grad()
            predictions = self.forward(state)

            loss = self.loss_fn(predictions, action.long())
            loss.backward()
            accumulated_loss.append(loss.item())
            self.optimizer_fn.step()

            accuracy: Number = None
            if self.discrete:
                predictions_argmax = torch.argmax(predictions, 1)
                accuracy = ((predictions_argmax == action).sum().item() / action.size(0)) * 100
            else:
                accuracy = (action - predictions).pow(2).sum(1).sqrt().mean().item()
            accumulated_accuracy.append(accuracy)

        metrics = {
            "idm_loss": np.mean(idm_accumulated_loss),
            "idm_accuracy": np.mean(idm_accumulated_accuracy),
            "loss": np.mean(accumulated_loss),
            "accuracy": np.mean(accumulated_accuracy)
        }
        return metrics

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
