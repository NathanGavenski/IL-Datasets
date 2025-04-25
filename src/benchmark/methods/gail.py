"""Module for GAIL method."""
from collections import defaultdict
from numbers import Number
import os

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from gymnasium import Env
from tensorboard_wrapper import Tensorboard
from tqdm import tqdm

from imitation_datasets.dataset.metrics import accuracy as accuracy_fn
from .method import Method, Metrics
from .utils import import_hyperparameters

PATH = "/".join(__file__.split("/")[:-1])
CONFIG_FILE = f"{PATH}/config/gail.yaml"


class Discriminator(nn.Module):
    """Discriminator network for GAIL."""

    def __init__(self, input_size: int, hidden_size: int = 256) -> None:
        """Initialize Discriminator network.

        Args:
            input_size (int): Size of the input tensor.
            hidden_size (int, optional): Size of the hidden layer. Defaults to 256.
        """
        super().__init__()

        self.state_encoder = None
        if isinstance(input_size, tuple):
            self.state_encoder = nn.Sequential(
                nn.Conv2d(input_size[0][-1], 32, kernel_size=8, stride=4),
                nn.LeakyReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.LeakyReLU(), 
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.LeakyReLU(),
                nn.Flatten(),
            )

            with torch.no_grad():
                state = torch.zeros(1, *input_size[0][::-1])
                n_flatten = self.state_encoder(state).size(1)

            input_size = n_flatten + input_size[1]

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Discriminator network.

        Args:
            state (torch.Tensor): Input state tensor.
            action (torch.Tensor): Input action tensor.

        Returns:
            torch.Tensor: Discriminator output tensor. 0 for policy actions and 1 for expert actions.
        """
        if self.state_encoder is not None:
            state = self.state_encoder(state)

        input_tensor = torch.cat([state, action], dim=-1)
        return self.net(input_tensor)


class GAIL(Method):
    """Generative Adversarial Imitation Learning method from (Ho et.al)."""

    __version__ = "1.0.0"
    __author__ = "Ho et. al."
    __method_name__ = "GAIL"

    def __init__(
        self,
        environment: Env,
        enjoy_criteria: int = 100,
        verbose: bool = False,
        config_file: str = None,
    ) -> None:
        """Initialize GAIL method.

        Args:
            environment (Env): The environment to train the model.
            enjoy_criteria (int, optional): Delta epochs to enjoy the model. Defaults to 100.
            verbose (bool, optional): Verbosity of the model. Defaults to False.
            config_file (str, optional): Configuration file for the hyperparameters. Defaults to None.
        """
        self.enjoy_criteria = enjoy_criteria
        self.verbose = verbose

        try:
            self.environment_name = environment.spec.name
        except AttributeError:
            self.environment_name = environment.spec._env_name
        self.save_path = f"./tmp/gail/{self.environment_name}"

        if config_file is None:
            config_file = CONFIG_FILE

        self.hyperparameters = import_hyperparameters(
            config_file,
            self.environment_name,
        )

        super().__init__(environment, self.hyperparameters)

        if self.visual:
            self.discriminator = Discriminator(input_size=(self.observation_size, self.action_size))
        else:
            self.discriminator = Discriminator(input_size=self.observation_size + self.action_size)

        self.discriminator.to(self.device)
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=self.hyperparameters.get("discriminator_lr", 5e4),
        )
        self.bce_loss = nn.BCELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.policy(x)

    def save(self, path: str = None, name: str = None) -> None:
        """Save the model.

        Args:
            path (str, optional): Path to save the model. Defaults to None.
                If None, save in the default path (self.save_path).
            name (str, optional): Name of the model. Defaults to None.
                If None, save with the default name (best_model.ckpt).
        """
        path = self.save_path if path is None else path
        if not os.path.exists(path):
            os.makedirs(path)

        name = "best_model.ckpt" if name is None else f"{name}.ckpt"

        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "discriminator_state_dict": self.discriminator.state_dict(),
            },
            f"{path}/{name}",
        )

    def load(self, path: str = None, name: str = None) -> Self:
        """Load the model.

        Args:
            path (str, optional): Path to load the model. Defaults to None.
                If None, load from the default path (self.save_path).
            name (str, optional): Name of the model. Defaults to None.
                If None, load the best model (best_model.ckpt).

        Returns:
            Self: The model loaded.
        """
        path = self.save_path if path is None else path
        name = "best_model" if name is None else name

        checkpoint = torch.load(
            f"{path}/{name}.ckpt", map_location=torch.device(self.device)
        )
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        return self

    def train(
        self,
        n_epochs: int,
        train_dataset: DataLoader,
        eval_dataset: DataLoader = None,
        always_save: bool = False,
        folder: str = None,
    ) -> Self:
        """Train the model using the GAIL algorithm.

        Args:
            n_epochs (int): Number of epochs to train the model.
            train_dataset (DataLoader): The dataset to train the model.
            eval_dataset (DataLoader, optional): The dataset to evaluate the model. Defaults to None.
            always_save (bool, optional): Save the model at each epoch. Defaults to False.
            folder (str, optional): The folder to save the results. Defaults to None.

        Returns:
            Self: The trained model.
        """
        if folder is None:
            folder = f"./benchmark_results/gail/{self.environment_name}"
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        board = Tensorboard(path=folder)
        board.add_hparams(self.hyperparameters)
        self.policy.to(self.device)
        self.discriminator.to(self.device)

        best_model = -np.inf
        pbar = range(n_epochs)
        if self.verbose:
            pbar = tqdm(pbar, desc=self.__method_name__)
        for epoch in pbar:
            train_metrics = self._train(train_dataset)
            board.add_scalars("Train", epoch="train", **train_metrics)

            if eval_dataset is not None:
                eval_metrics = self._eval(eval_dataset)
                board.add_scalars("Eval", epoch="eval", **eval_metrics)
                board.step(["train", "eval"])
            else:
                board.step("train")

            if epoch > 0 and epoch % self.enjoy_criteria == 0:
                metrics = self._enjoy()
                board.add_scalars("Enjoy", epoch="enjoy", **metrics)
                board.step("enjoy")

                if best_model < metrics["aer"] or always_save:
                    best_model = metrics["aer"]
                    self.save(name=epoch if always_save else None)

                    if self.early_stop(best_model):
                        return self
        return self

    def _train(self, dataset: DataLoader) -> Metrics:
        """Train the model using the GAIL algorithm.

        Args:
            dataset (DataLoader): The dataset to train the model.

        Returns:
            Metrics: The metrics of the training.
        """
        self.policy.train()
        self.discriminator.train()

        metrics = defaultdict(list)
        for batch in dataset:
            state, action, _ = batch
            state = state.to(self.device)
            action = action.to(self.device)

            policy_actions = self.forward(state)
            accuracy: Number = None
            if self.discrete:
                accuracy = accuracy_fn(policy_actions, action.squeeze(1))
            else:
                accuracy = (action - policy_actions).pow(2).sum(1).sqrt().mean().item()
            metrics["policy_acc"].append(accuracy)

            # Train Discriminator
            if self.discrete:
                action = nn.functional.one_hot(
                    action.squeeze(1).long(),
                    self.action_size
                ).float()
            expert_labels = torch.ones((state.size(0), 1)).to(self.device)
            policy_labels = torch.zeros((state.size(0), 1)).to(self.device)
            self.discriminator_optimizer.zero_grad()
            expert_outputs = self.discriminator(state.float(), action)
            policy_outputs = self.discriminator(state.float(), policy_actions.detach())

            expert_acc = accuracy_fn(expert_outputs, expert_labels.squeeze(1))
            policy_acc = accuracy_fn(policy_outputs, policy_labels.squeeze(1))
            metrics["disc_acc"].append((policy_acc + expert_acc) / 2)

            expert_loss = self.bce_loss(expert_outputs, expert_labels)
            policy_loss = self.bce_loss(policy_outputs, policy_labels)

            loss = expert_loss + policy_loss
            loss.backward()
            self.discriminator_optimizer.step()
            metrics["disc_loss"].append(loss.item())

            # Train Policy
            self.optimizer_fn.zero_grad()
            policy_outputs = self.discriminator(state.float(), policy_actions)
            loss = self.bce_loss(policy_outputs, expert_labels)
            loss.backward()
            self.optimizer_fn.step()
            metrics["policy_loss"].append(loss.item())

        return {k: np.mean(v) for k, v in metrics.items()}

    def _eval(self, dataset: DataLoader) -> Metrics:
        """Evaluate the model using the GAIL algorithm.

        Args:
            dataset (DataLoader): The dataset to evaluate the model.

        Returns:
            Metrics: The metrics of the evaluation.
        """
        self.policy.eval()

        metrics = defaultdict(list)
        for batch in dataset:
            state, action, _ = batch
            state = state.to(self.device)

            with torch.no_grad():
                predictions = self.forward(state)

            accuracy: Number = None
            if self.discrete:
                accuracy = accuracy_fn(predictions, action.squeeze(1))
            else:
                accuracy = (action - predictions).pow(2).sum(1).sqrt().mean().item()
            metrics["accuracy"].append(accuracy)

        return {k: np.mean(v) for k, v in metrics.items()}

    def early_stop(self, metric: Metrics) -> bool:
        """Early stop criteria for the model.

        Args:
            metric (Metrics): The metric to evaluate the model.

        Returns:
            bool: True if the model should stop, False otherwise.
        """
        return False
