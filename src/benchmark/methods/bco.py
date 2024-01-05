"""Module for Behavioural Cloning from Observation"""
from collections import defaultdict
import os
from typing import List, Union, Dict, Tuple
from numbers import Number

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from gymnasium import Env
from tensorboard_wrapper.tensorboard import Tensorboard

from imitation_datasets.dataset.metrics import accuracy as accuracy_fn
from imitation_datasets.dataset.metrics import average_episodic_reward, performance
from imitation_datasets.utils import GymWrapper
from imitation_datasets.dataset import get_random_dataset, BaselineDataset
from .policies.mlp import MLP, MlpWithAttention
from .method import Metrics, Method
from .utils import import_hyperparameters


CONFIG_FILE = "./src/benchmark/methods/config/bco.yaml"


class BCO(Method):
    """Behavioural Cloning from Observation method based on (Torabi et. al., 2018)"""

    __version__ = "1.0.0"
    __author__ = "Torabi et. al."
    __method_name__ = "Behavioural Cloning from Observation"

    def __init__(self, environment: Env, enjoy_criteria: int = 100, verbose: bool = False) -> None:
        """Initialize BCO method."""
        self.enjoy_criteria = enjoy_criteria
        self.verbose = verbose
        self.environment_name = environment.spec.name
        self.save_path = f"./tmp/bco/{self.environment_name}/"

        self.hyperparameters = import_hyperparameters(
            CONFIG_FILE,
            environment.spec.id,
        )

        super().__init__(
            environment,
            self.hyperparameters
        )

        idm = self.hyperparameters.get('idm', 'MlpPolicy')
        if idm == 'MlpPolicy':
            self.idm = MLP(self.observation_size * 2, self.action_size)
        elif idm == 'MlpWithAttention':
            self.idm = MlpWithAttention(self.observation_size * 2, self.action_size)

        self.idm_optimizer = optim.Adam(self.idm.parameters(), lr=self.hyperparameters['idm_lr'])
        self.idm_loss = nn.CrossEntropyLoss() if self.discrete else nn.MSELoss()

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
        return self

    def train(
        self,
        n_epochs: int,
        train_dataset: Dict[str, DataLoader],
        eval_dataset: Dict[str, DataLoader] = None,
        folder: str = None
    ) -> Self:
        """Train process.

        Args:
            n_epochs (int): amount of epoch to run.
            train_dataset (DataLoader): data to train.
            eval_dataset (DataLoader): data to eval. Defaults to None.

        Returns:
            method (Self): trained method.
        """
        if folder is None:
            folder = f"../benchmark_results/bco/{self.environment_name}"
        if not os.path.exists(folder):
            os.makedirs(f"{folder}/")
        board = Tensorboard(path=folder)
        self.policy.to(self.device)
        self.idm.to(self.device)

        best_model = -np.inf
        if not isinstance(train_dataset, dict):
            train_dataset = {"expert_dataset": train_dataset}

        if "idm_dataset" not in train_dataset.keys():
            print("No random dataset found")
            random_path = f"./dataset/random_{self.environment.spec.id}"

            if not os.path.exists(random_path):
                print("Creating random dataset from scratch")
                train_dataset["idm_dataset"] = get_random_dataset(
                    environment_name=self.environment.spec.id,
                    episodes=self.hyperparameters["random_episodes"]
                )
            else:
                print("Loading local random dataset")
                train_dataset["idm_dataset"] = BaselineDataset(
                    f"{random_path}/teacher.npz"
                )

            train_dataset["idm_dataset"] = DataLoader(
                train_dataset["idm_dataset"],
                batch_size=train_dataset["expert_dataset"].batch_size,
                shuffle=True
            )

        pbar = range(n_epochs)
        if self.verbose:
            pbar = tqdm(pbar)
        for epoch in pbar:
            train_metrics = self._train(**train_dataset)
            board.add_scalars("Train", epoch="train", **train_metrics)

            if eval_dataset is not None:
                eval_metrics = self._eval(eval_dataset)
                board.add_scalars("Eval", epoch="eval", **eval_metrics)
                board.step(["train", "eval"])
            else:
                board.step("train")

            if epoch % self.enjoy_criteria == 0:
                train_dataset = self._append_samples(train_dataset)

            if epoch % self.enjoy_criteria == 0 or epoch + 1 == n_epochs:
                metrics = self._enjoy()
                board.add_scalars("Enjoy", epoch="enjoy", **metrics)
                board.step("enjoy")
                if best_model < metrics["aer"]:
                    self.save()

        return self

    def _append_samples(self, train_dataset: DataLoader) -> DataLoader:
        """Append samples to DataLoader.

        Args:
            train_dataset (DataLoader): current train dataset.

        Returns:
            train_dataset (DataLoader): new train dataset.
        """
        _, i_pos = self._enjoy(return_ipos=True)
        train_dataset['idm_dataset'].dataset.states = torch.cat((
            train_dataset['idm_dataset'].dataset.states,
            torch.from_numpy(i_pos['states'])),
            dim=0
        )
        train_dataset['idm_dataset'].dataset.next_states = torch.cat((
            train_dataset['idm_dataset'].dataset.next_states,
            torch.from_numpy(i_pos['next_states'])),
            dim=0
        )
        train_dataset['idm_dataset'].dataset.actions = torch.cat((
            train_dataset['idm_dataset'].dataset.actions,
            torch.from_numpy(i_pos['actions'].reshape((-1, 1)))),
            dim=0
        )
        return train_dataset

    def _train(self, idm_dataset: DataLoader, expert_dataset: DataLoader) -> Metrics:
        """Train loop.

        Args:
            dataset (DataLoader): train data.
        """
        if not self.idm.training:
            self.idm.train()

        if not self.policy.training:
            self.policy.train()

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

            loss = self.idm_loss(predictions, action.squeeze(1).long())
            loss.backward()
            idm_accumulated_loss.append(loss.item())
            self.idm_optimizer.step()

            accuracy: Number = None
            if self.discrete:
                accuracy = accuracy_fn(predictions, action.squeeze(1))
            else:
                accuracy = (action - predictions).pow(2).sum(1).sqrt().mean().item()
            idm_accumulated_accuracy.append(accuracy)

        self.idm.eval()

        for batch in expert_dataset:
            state, _, next_state = batch
            state = state.to(self.device)
            next_state = next_state.to(self.device)

            with torch.no_grad():
                if self.discrete:
                    action = self.idm(torch.cat((state, next_state), dim=1))
                    action = torch.argmax(action, dim=1)
                else:
                    action = self.idm(torch.cat((state, next_state), dim=1))

            self.optimizer_fn.zero_grad()
            predictions = self.forward(state)

            loss = self.loss_fn(predictions, action.squeeze(1).long())
            loss.backward()
            accumulated_loss.append(loss.item())
            self.optimizer_fn.step()

            accuracy: Number = None
            if self.discrete:
                accuracy = accuracy_fn(predictions, action.squeeze(1))
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
        if self.policy.training:
            self.policy.eval()

        accumulated_accuracy = []

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

    def _enjoy(
        self,
        render: bool = False,
        teacher_reward: Number = None,
        random_reward: Number = None,
        return_ipos: bool = False,
    ) -> Union[Metrics, Tuple[Metrics, Dict[str, List[float]]]]:
        """Function for evaluation of the policy in the environment

        Args:
            render (bool): Whether it should render. Defaults to False.
            teacher_reward (Number): reward for teacher policy.
            random_reward (Number): reward for a random policy.
            return_ipos (bool): whether it should return data to append to I_pos.

        Returns:
            Metrics:
                aer (Number): average reward for 100 episodes.
                aer_std (Number): standard deviation for aer.
                performance (Number): if teacher_reward and random_reward are
                    informed than the performance metric is calculated.
                perforamance_std (Number): standard deviation for performance.
            I_pos:
                states (List[Number]): states before action.
                actions (List[Number]): action given states.
                next_states (List[Number]): next state given states and actions.
        """
        environment = GymWrapper(self.environment)
        average_reward = []
        i_pos = defaultdict(list)

        for _ in range(100):
            done = False
            obs = environment.reset()
            accumulated_reward = 0
            while not done:
                if render:
                    environment.render()
                action = self.predict(obs)

                i_pos['states'].append(obs)
                i_pos['actions'].append(action)

                obs, reward, done, *_ = environment.step(action)
                accumulated_reward += reward
                i_pos['next_states'].append(obs)
            average_reward.append(accumulated_reward)

        metrics = average_episodic_reward(average_reward)
        if teacher_reward is not None and random_reward is not None:
            metrics.update(performance(average_reward, teacher_reward, random_reward))

        i_pos = {key: np.array(value) for key, value in i_pos.items()}

        if return_ipos:
            return metrics, i_pos
        return metrics
