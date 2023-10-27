"""Module for Behavioural Cloning"""
import os
from typing import List, Union
from numbers import Number
from datetime import date

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import torch
from torch.utils.data import DataLoader
from gymnasium import Env, spaces
from tensorboard_wrapper.tensorboard import Tensorboard
from tqdm import tqdm

from .policies.mlp import MLP
from .method import Method, Metrics
from imitation_datasets.dataset.metrics import accuracy as accuracy_fn


# TODO adapt for visual
class BC(Method):
    """Behavioural Clonning method based on (Pomerleau, 1988)"""

    def __init__(self, environment: Env, enjoy_criteria: int = 10, verbose: bool = False) -> None:
        """Initialize BC method."""
        self.enjoy_criteria = enjoy_criteria
        self.verbose = verbose
        self.environment = environment
        self.discrete = isinstance(environment.action_space, spaces.Discrete)
        self.observation_size = environment.observation_space.shape[0]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.action_size = None
        if self.discrete:
            self.action_size = environment.action_space.n
        else:
            self.action_size = environment.action_space.shape[0]

        self.save_path = "./tmp/bc/"

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
        """Save all model weights.

        Args:
            path (str): where to save the models. Defaults to None.
        """
        path = self.save_path if path is None else path
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.policy.state_dict(), f"{path}/best_model.ckpt")

    def load(self, path: str = None) -> None:
        """Load all model weights.

        Args:
            path (str): where to look for the model's weights. Defaults to None.

        Raises:
            ValueError: if the path does not exist.
        """
        path = self.save_path if path is None else path

        if not os.path.exists(path):
            raise ValueError("Path does not exists.")

        self.policy.load_state_dict(torch.load(f"{path}best_model.ckpt"))

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
        folder = "./benchmark_results/bc/"
        if not os.path.exists(folder):
            os.makedirs(f"{folder}/")
        board = Tensorboard(path=folder, name=date.today(), delete=True)

        best_model = -np.inf

        pbar = tqdm(range(n_epochs)) if self.verbose else range(n_epochs)
        for epoch in pbar:
            train_metrics = self._train(train_dataset)
            board.add_scalars("Train", epoch="train", **train_metrics)

            if eval_dataset is not None:
                eval_metrics = self._eval(eval_dataset)
                board.add_scalars("Eval", epoch="eval", **eval_metrics)
            board.step("train")
            board.step("eval")

            if epoch % self.enjoy_criteria == 0:
                average_reward = self._enjoy()
                board.add_scalars("Enjoy", epoch="enjoy", reward=average_reward)
                board.step("enjoy")
                if best_model < average_reward:
                    self.save()

        return self

    def _train(self, dataset: DataLoader) -> Metrics:
        """Train loop.

        Args:
            dataset (DataLoader): train data.
        """
        accumulated_loss = []
        accumulated_accuracy = []

        if not self.policy.training:
            self.policy.train()

        for batch in dataset:
            state, action, _ = batch
            state = state.to(self.device)
            action = action.to(self.device)

            self.optimizer_fn.zero_grad()
            predictions = self.forward(state)

            loss = self.loss_fn(predictions, action.squeeze().long())
            loss.backward()
            accumulated_loss.append(loss.item())
            self.optimizer_fn.step()

            accuracy: Number = None
            if self.discrete:
                accuracy = accuracy_fn(predictions, action.squeeze())
            else:
                accuracy = (action - predictions).pow(2).sum(1).sqrt().mean().item()
            accumulated_accuracy.append(accuracy)

        return {"loss": np.mean(accumulated_loss), "accuracy": np.mean(accumulated_accuracy)}

    def _eval(self, dataset: DataLoader) -> Metrics:
        """Evaluation loop.

        Args:
            dataset (DataLoader): data to eval.
        """
        accumulated_accuracy = []

        if self.policy.training:
            self.policy.eval()

        for batch in dataset:
            state, action, _ = batch
            state = state.to(self.device)

            with torch.no_grad:
                predictions = self.policy(state)

            accuracy: Number = None
            if self.discrete:
                predictions_argmax = torch.argmax(predictions, 1)
                accuracy = ((predictions_argmax == action).sum().item() / action.size(0)) * 100
            else:
                accuracy = (action - predictions).pow(2).sum(1).sqrt().mean().item()
            accumulated_accuracy.append(accuracy)

        return {"accuracy": np.mean(accumulated_accuracy)}
