"""Module for DAgger method."""
from collections import defaultdict
from numbers import Number
import os
from typing import Union, List

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from gymnasium import Env
import numpy as np
from tensorboard_wrapper import Tensorboard
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from .bc import BC
from .utils import import_hyperparameters
from imitation_datasets.utils import GymWrapper
from imitation_datasets import Policy


PATH = "/".join(__file__.split("/")[:-1])
CONFIG_FILE = f"{PATH}/config/dagger.yaml"


class DAgger(BC):
    """DAgger method based on (Ross et. al., 2010)"""

    __version__ = "1.0.0"
    __author__ = "Ross et. al."
    __method_name__ = "DAgger"

    def __init__(
        self,
        environment: Env,
        enjoy_criteria: int = 100,
        verbose: bool = False,
        config_file: str = None
    ) -> None:
        self.enjoy_criteria = enjoy_criteria
        self.verbose = verbose
        self.expert: Policy = None

        try:
            self.environment_name = environment.spec.name
        except AttributeError:
            self.environment_name = environment.spec._env_name
        self.save_path = f"./tmp/DAgger/{self.environment_name}"

        if config_file is None:
            config_file = CONFIG_FILE

        self.hyperparameters = import_hyperparameters(
            config_file,
            self.environment_name
        )

        super().__init__(
            environment,
            enjoy_criteria,
            verbose,
            config_file
        )

    def train(
        self,
        n_epochs: int,
        train_dataset: DataLoader,
        expert: Policy,
        eval_dataset: DataLoader = None,
        always_save: bool = False,
        folder: str = None,
    ) -> Self:
        """Train the model using the DAgger algorithm.

        Args:
            n_epochs (int): Number of epochs to train the model.
            train_dataset (DataLoader): The dataset to train the model.
            expert (Policy): The expert policy to collect data.
            eval_dataset (DataLoader, optional): The dataset to evaluate the model. Defaults to None.
            always_save (bool, optional): Save the model at each epoch. Defaults to False.
            folder (str, optional): The folder to save the results. Defaults to None.

        Returns:
            Self: The trained model.
        """
        if folder is None:
            folder = f"./benchmark_results/dagger/{self.environment_name}"
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        board = Tensorboard(path=folder)
        board.add_hparams(self.hyperparameters)
        self.policy.to(self.device)

        self.expert = expert
        self.dataset = train_dataset.dataset

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

            self._collect_data()

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

    def _collect_data(
        self,
    ) -> None:
        """Collect data using the expert policy.

        Args:
            dataset (BaselineDataset): The dataset to collect data.
            transforms (Callable[[np.ndarray], torch.Tensor], optional): The transforms to apply to the observations. Defaults to None.

        Returns:
            BaselineDataset: The updated dataset.
        """
        environment = GymWrapper(self.environment, "newest")
        new_data = defaultdict(list)

        for _ in range(self.hyperparameters.get("n_rollouts", 10)):
            obs = environment.reset()
            done = False
            while not done:
                expert_action = self._get_expert_action(obs)
                next_obs, _, done, *_ = environment.step(expert_action)

                new_data["obs"].append(obs)
                new_data["action"].append(expert_action)
                new_data["next_obs"].append(next_obs)  # Mostly to satisfy the dataset
        new_data["obs"] = torch.from_numpy(np.array(new_data["obs"]))
        new_data["action"] = torch.from_numpy(np.array(new_data["action"])).unsqueeze(1)
        new_data["next_obs"] = torch.from_numpy(np.array(new_data["next_obs"]))

        self.dataset.states = torch.cat((self.dataset.states, new_data["obs"]), dim=0)
        self.dataset.actions = torch.cat((self.dataset.actions, new_data["action"]), dim=0)
        self.dataset.next_states = torch.cat((self.dataset.next_states, new_data["next_obs"]), dim=0)

    def _get_expert_action(self, obs: np.ndarray) -> Union[List[Number], Number]:
        """Get the expert action for the given observation.

        Args:
            obs (np.ndarray): The observation.

        Returns:
            Union[List[Number], Number]: The expert action.
        """
        expert_action, _ = self.expert.predict(obs, deterministic=True)
        return expert_action
