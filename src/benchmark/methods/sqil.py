from copy import deepcopy
from collections import deque
from itertools import count
from typing import Union, List, Callable
from numbers import Number
import os

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from gymnasium import Env
import numpy as np
from tensorboard_wrapper import Tensorboard
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.distributions import Categorical

from imitation_datasets.dataset.metrics import accuracy as accuracy_fn
from .method import Method, Metrics
from .utils import import_hyperparameters, ReplayBuffer


PATH = "/".join(__file__.split("/")[:-1])
CONFIG_FILE = f"{PATH}/config/sqil.yaml"


class SQIL(Method):
    """Soft-Q Imitation Learning method based on ()."""

    __version__ = "1.0.0"
    __author__ = ""
    __method_name__ = "Soft-Q Imitation Learning"

    def __init__(
        self,
        environment: Env,
        enjoy_criteria: int = 100,
        verbose: bool = False,
        config_file: str = None,
    ) -> None:
        self.enjoy_criteria = enjoy_criteria
        self.verbose = verbose
        try:
            self.environment_name = environment.spec.name
        except AttributeError:
            self.environment_name = environment.spec._env_name
        self.save_path = f"./tmp/sqil/{self.environment_name}"

        if config_file is None:
            config_file = CONFIG_FILE

        self.hyperparameters = import_hyperparameters(
            config_file,
            self.environment_name,
        )

        super().__init__(environment, self.hyperparameters)
        self.target: nn.Module = deepcopy(self.policy)
        self.target.load_state_dict(self.policy.state_dict())
        self.loss_fn = nn.functional.mse_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return self.policy(x)

    def get_value(self, q_value) -> torch.Tensor:
        alpha = self.hyperparameters.get("alpha", 4)
        value = alpha * torch.log(torch.sum(torch.exp(q_value / alpha), dim=1, keepdim=True))
        return value

    def predict(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        transforms: Callable[[torch.Tensor], torch.Tensor] = None,
        enjoy: bool = True
    ) -> Union[List[Number], Number]:
        if enjoy:
            self.policy.eval()

        if isinstance(obs, np.ndarray):
            if not self.visual:
                obs = torch.from_numpy(obs)
                if transforms is not None:
                    obs = transforms(obs)
                if len(obs.shape) == 1:
                    obs = obs[None]
            else:
                if transforms is None:
                    raise ValueError("Visual information requires transforms parameter.")
                obs = transforms(obs)
                if len(obs.shape) == 3:
                    obs = obs[None]

        obs = obs.to(self.device)

        with torch.no_grad():
            q = self.forward(obs)
            v = self.get_value(q)
            dist = torch.exp((q - v) / self.hyperparameters.get("alpha", 4))
            dist = dist / torch.sum(dist)
            action = Categorical(dist).sample()
        return action.item()


    def save(self, path: str = None, name: str = None) -> None:
        path = self.save_path if path is None else path
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        name = "best_model.ckpt" if name is None else f"{name}.ckpt"
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "target_state_dict": self.target.state_dict(),
            },
            f"{path}/{name}",
        )

    def load(self, path: str = None, name: str = None) -> Self:
        path = self.save_path if path is None else path
        name = "best_model" if name is None else name

        checkpoint = torch.load(
            f"{path}/{name}.ckpt", map_location=torch.device(self.device)
        )
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.target.load_state_dict(checkpoint["target_state_dict"])
        return self

    def train(
        self,
        n_epochs: int,
        train_dataset: DataLoader,
        eval_dataset: DataLoader = None,
        always_save: bool = False,
        folder: str = None,
    ) -> Self:
        if folder is None:
            folder = f"./benchmark_results/sqil/{self.environment_name}"
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        board = Tensorboard(path=folder)
        board.add_hparams(self.hyperparameters)
        self.policy.to(self.device)
        self.target.to(self.device)

        # Since it trains online we have to simulate the number of epochs by
        # computing the number of updates and replicating it
        n_epochs = len(train_dataset) * n_epochs

        self.replay_buffer = ReplayBuffer(
            buffer_size=self.hyperparameters.get("buffer_size", 50000),
            batch_size=train_dataset.batch_size
        )
        self.replay_buffer.load(train_dataset.dataset)
        self._collect_samples()

        best_model = -np.inf
        pbar = range(n_epochs)
        if self.verbose:
            pbar = tqdm(pbar, desc=self.__method_name__)
        
        self.obs = None
        for epoch in pbar:
            self.obs = self._environment_step(self.obs)
            metrics = self._train()
            board.add_scalars("Train", epoch="train", **metrics)

            if eval_dataset is not None:
                metrics = self._eval(eval_dataset)
                board.add_scalars("Eval", epoch="eval", **metrics)
                board.step(["train", "eval"])
            else:
                board.step("train")

            if epoch > 0 and self.hyperparameters.get("update_steps", 4) % epoch == 0:
                self.target.load_state_dict(self.policy.state_dict())

            if epoch > 0 and epoch % self.enjoy_criteria == 0:
                self.tmp_environment = deepcopy(self.environment)
                metrics = self._enjoy()
                self.environment = self.tmp_environment
                board.add_scalars("Enjoy", epoch="enjoy", **metrics)
                board.step("enjoy")
                if best_model < metrics["aer"] or always_save:
                    best_model = metrics["aer"]
                    self.save(name=epoch if always_save else None)

                    if self.early_stop(metrics["aer"]):
                        return self
        return self

    def _collect_samples(self) -> None:
        obs = None
        while self.replay_buffer.size() < self.hyperparameters.get("buffer_size", 50000):
            obs = self._environment_step(obs)

    def _environment_step(self, obs: np.ndarray) -> np.ndarray:
        if obs is None:
            obs, _ = self.environment.reset()
        action = self.predict(obs)
        next_obs, reward, done, terminated, _ = self.environment.step(action)
        done |= terminated
        self.replay_buffer.add((
            torch.from_numpy(obs),
            torch.from_numpy(next_obs),
            torch.tensor([action]),
            torch.tensor(0.),
            torch.tensor(done)
        ))

        obs = next_obs
        if done:
            obs, _ = self.environment.reset()
        return obs

    def _train(self) -> Metrics:
        """Train loop.

        Args:
            dataset (DataLoader): train data.
        """
        if not self.policy.training:
            self.policy.train()

        policy_data, expert_data = self.replay_buffer.sample()
        policy_state, policy_next_state, policy_action, policy_reward, policy_done = policy_data
        policy_state, policy_next_state, policy_action, policy_reward, policy_done = map(
            lambda x: x.to(self.device), (x for x in policy_data)
        )

        expert_state, expert_action, expert_next_state, expert_reward, expert_done = expert_data
        expert_state, expert_next_state, expert_action, expert_reward, expert_done = map(
            lambda x: x.to(self.device), (
                expert_state, expert_next_state, expert_action, expert_reward, expert_done
            )
        )

        batch_state = torch.cat([policy_state, expert_state], dim=0)
        batch_next_state = torch.cat([policy_next_state, expert_next_state], dim=0)
        batch_action = torch.cat([policy_action, expert_action], dim=0)
        batch_reward = torch.cat([policy_reward, expert_reward], dim=0).unsqueeze(1)
        batch_done = torch.cat([policy_done, expert_done], dim=0).unsqueeze(1)

        with torch.no_grad():
            next_q = self.target(batch_next_state)
            next_v = self.get_value(next_q)
            gamma = self.hyperparameters.get("gamma", 0.99)
            y = batch_reward + (1 - batch_done) * gamma + next_v

        self.optimizer_fn.zero_grad()
        predictions = self.forward(batch_state).gather(1, batch_action.long())
        loss = self.loss_fn(predictions, y.float())
        loss.backward()
        self.optimizer_fn.step()
        return {"loss": loss.item()}

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
                actions = self.predict(state)

            accuracy: Number = None
            if self.discrete:
                accuracy = accuracy_fn(actions, action.squeeze(1))
            else:
                accuracy = (action - actions).pow(2).sum(1).sqrt().mean().item()
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
