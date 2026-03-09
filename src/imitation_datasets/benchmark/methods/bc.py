"""Module for Behavioural Cloning"""
import os
from numbers import Number
from typing import Callable

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from gymnasium import Env
import numpy as np
import torch
from torch.utils.data import DataLoader
from tensorboard_wrapper.tensorboard import Tensorboard
from tqdm import tqdm

from imitation_datasets.dataset.metrics import accuracy as accuracy_fn
from .method import Method, Metrics
from .utils import import_hyperparameters


PATH = "/".join(__file__.split("/")[:-1])
CONFIG_FILE = f"{PATH}/config/bc.yaml"


class BC(Method):
    """Behavioural Clonning method based on (Pomerleau, 1988)"""

    __version__ = "1.0.0"
    __author__ = "Pomerleau"
    __method_name__ = "Behavioural Cloning"

    def __init__(
        self,
        environment: Env,
        enjoy_criteria: int = 100,
        verbose: bool = False,
        config_file: str = None
    ) -> None:
        """Initialize BC method."""
        self.enjoy_criteria = enjoy_criteria
        self.verbose = verbose
        try:
            self.environment_name = environment.spec.name
        except AttributeError:
            self.environment_name = environment.spec._env_name
        self.save_path = f"./tmp/bc/{self.environment_name}/"

        if config_file is None:
            config_file = CONFIG_FILE

        self.hyperparameters = import_hyperparameters(
            config_file,
            self.environment_name,
        )

        super().__init__(
            environment,
            self.hyperparameters
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for the method.

        Args:
            x (torch.Tensor): input.

        Returns:
            x (torch.Tensor): logits output.
        """
        return self.policy(x)

    def save(self, path: str = None, name: str = None) -> None:
        """Save all model weights.

        Args:
            path (str): where to save the models. Defaults to None.
        """
        path = self.save_path if path is None else path
        if not os.path.exists(path):
            os.makedirs(path)

        name = "best_model.ckpt" if name is None else f"{name}.ckpt"

        torch.save(self.policy.state_dict(), f"{path}/{name}")

    def load(self, path: str = None, name: str = None) -> Self:
        """Load all model weights.

        Args:
            path (str): where to look for the model's weights. Defaults to None.

        Raises:
            ValueError: if the path does not exist.
        """
        path = self.save_path if path is None else path
        name = "best_model" if name is None else name

        if not os.path.exists(path):
            raise ValueError("Path does not exists.")

        self.policy.load_state_dict(
            torch.load(
                f"{path}{name}.ckpt",
                map_location=torch.device(self.device)
            )
        )

        return self

    def train(
        self,
        n_epochs: int,
        train_dataset: DataLoader,
        eval_dataset: DataLoader = None,
        always_save: bool = False,
        folder: str = None
    ) -> Self:
        """Train process.

        Args:
            n_epochs (int): amount of epoch to run.
            train_dataset (DataLoader): data to train.
            eval_dataset (DataLoader): data to eval. Defaults to None.
            always_save (bool): whether it should save all eval steps.
            folder (str): a specific folder to save the benchmark results.

        Returns:
            method (Self): trained method.
        """
        if folder is None:
            folder = f"./benchmark_results/bc/{self.environment_name}"
        if not os.path.exists(folder):
            os.makedirs(f"{folder}/")

        board = Tensorboard(path=folder)
        board.add_hparams(self.hyperparameters)
        self.policy.to(self.device)

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

                    if self.early_stop(metrics["aer"]):
                        return self

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

            loss = self.loss_fn(predictions, action.squeeze(1).long())
            loss.backward()
            self.optimizer_fn.step()
            accumulated_loss.append(loss.item())

            accuracy: Number = None
            if self.discrete:
                accuracy = accuracy_fn(predictions, action.squeeze(1))
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

            with torch.no_grad():
                predictions = self.policy(state)

            accuracy: Number = None
            if self.discrete:
                accuracy = accuracy_fn(predictions, action.squeeze(1))
            else:
                accuracy = (action - predictions).pow(2).sum(1).sqrt().mean().item()
            accumulated_accuracy.append(accuracy)

        return {"accuracy": np.mean(accumulated_accuracy)}

    def early_stop(self, metric: Metrics) -> bool:
        """Function that tells the method if it should stop or not.

        Args:
            metric (Metrics): the metric to evaluate whether it should early stop.

        Returns:
            stop (bool): if it should stop or not.
        """
        return False
