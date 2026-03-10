from collections import defaultdict
import copy
from functools import partial
from numbers import Number
import os
from typing import Tuple, Dict, Any, Callable, Union
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from imitation_datasets.baselines.methods.method import Method, Metrics
from imitation_datasets.baselines.methods.method import MLP, CNN, Resnet
from imitation_datasets.baselines.methods.utils import import_hyperparameters
from gymnasium import Env
import numpy as np
from tensorboard_wrapper import Tensorboard
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


PATH = "/".join(__file__.split("/")[:-1])
CONFIG_FILE = f"{PATH}/config/offline_iqlearn.yaml"


class OfflineIQLearn(Method):
    __version__ = "1.0.0"
    __author__ = "Garg et. al."
    __method_name__ = "Inverse Q-Learn"

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
        self.save_path = f"./tmp/iqlearn/{self.environment_name}"

        if config_file is None:
            config_file = CONFIG_FILE

        self.hyperparameters = import_hyperparameters(
            config_file,
            self.environment_name
        )

        super().__init__(environment, self.hyperparameters, activation=nn.ELU)

        self.q_target = copy.deepcopy(self.policy)
        self.q_target.to(self.device)
        self.q_target.load_state_dict(self.policy.state_dict())

        self.actor_optimizer = None
        self.log_alpha = torch.tensor(np.log(self.hyperparameters.get("alpha", 0.5))).to(self.device)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_value(self, obs, actions=None):
        q = self.policy(obs)
        v = self.alpha * torch.logsumexp(q / self.alpha, dim=1, keepdim=True)
        return v

    def get_targetV(self, obs):
        q = self.q_target(obs)
        v = self.alpha * torch.logsumexp(q / self.alpha, dim=1, keepdim=True)
        return v

    def save(self, path: str = None, name: str = None) -> None:
        path = self.save_path if path is None else path
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        if self.policy:
            _name = "best_q_network.ckpt" if name is None else f"{name}_q_network.ckpt"
            torch.save(self.policy.state_dict(), f"{path}/{_name}")

        if self.q_target:
            _name = "best_q_target.ckpt" if name is None else f"{name}_q_target.ckpt"
            torch.save(self.q_target.state_dict(), f"{path}/{_name}")

    def load(self, path: str = None, name: str = None) -> Self:
        path = self.save_path if path is None else path
        name = "best" if name is None else name

        if not os.path.exists(path):
            raise ValueError("Path does not exist.")

        if self.policy:
            state_dict = torch.load(
                f"{path}{name}_q_network.ckpt",
                map_location=torch.device(self.device)
            )
            self.policy.load_state_dict(state_dict)

        if self.q_target:
            state_dict = torch.load(
                f"{path}{name}_q_target.ckpt",
                map_location=torch.device(self.device)
            )
            self.q_target.load_state_dict(state_dict)

        return self

    @torch.no_grad()
    def predict(
        self,
        obs: Union[torch.Tensor, np.ndarray],
        transforms: Callable[[torch.Tensor], torch.Tensor] = None,
        deterministic: bool = True
    ) -> Number:
        self.policy.eval()
        if isinstance(obs, np.ndarray) and transforms is None:
            obs = torch.from_numpy(obs)

        if transforms is not None:
            obs = transforms(obs)

        obs = obs.to(self.device)

        if len(obs.shape) in [1, 3]:
            obs = obs[None]

        q_values = self.policy(obs)
        if deterministic:
            action = q_values.argmax(dim=1)
        else:
            probs = F.softmax(q_values / self.hyperparameters.get("temperature"), dim=1)
            action = torch.multinomial(probs, 1)
        return action.cpu().numpy()[0]

    def train(
        self,
        n_epochs: int,
        train_dataset: DataLoader,
        eval_dataset: DataLoader = None,
        transforms: Callable[[torch.Tensor], torch.Tensor] = None,
        always_save: bool = True,
        folder: str = None
    ) -> Self:
        if folder is None:
            folder = f"./benchmark_results/iqlearn/{self.environment_name}"
        if not os.path.exists(folder):
            os.makedirs(f"{folder}/", exist_ok=True)

        board = Tensorboard(path=folder, delete=True)
        
        hparams = {}
        for key, value in self.hyperparameters.items():
            if isinstance(value, list):
                value = ",".join([str(v) for v in value])
            hparams[key] = value
        board.add_hparams(hparams)
        self.policy.to(self.device)
        self.q_target.to(self.device)

        best_model = -np.inf

        pbar = range(n_epochs)
        if self.verbose:
            pbar = tqdm(pbar, desc=self.__method_name__)
        for epoch in pbar:
            train_metrics = self.loop(train_dataset, train=True)
            board.add_scalars("Train", epoch="train", **train_metrics)

            if eval_dataset is not None:
                eval_metrics = self.loop(eval_dataset, train=False)
                board.add_scalars("Eval", epoch="eval", **eval_metrics)
                board.step(["train", "eval"])
            else:
                board.step("train")

            if epoch % self.hyperparameters.get("update_frequency", 4) == 0:
                tau = self.hyperparameters.get("tau", 0.05)
                for target_param, param in zip(self.q_target.parameters(), self.policy.parameters()):
                    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

            if epoch > 0 and epoch % self.enjoy_criteria == 0:
                metrics = self._enjoy(transforms=transforms)
                board.add_scalars("Enjoy", epoch="enjoy", **metrics)
                board.step("enjoy")

                if best_model < metrics["aer"] or always_save:
                    if best_model < metrics["aer"]:
                        self.save()
                        best_model = metrics["aer"]

                    if always_save:
                        self.save(name=epoch)

                    if self.early_stop(metrics["aer"]):
                        return self
        return self

    def early_stop(self, metric: Metrics) -> bool:
        return False

    def loop(self, dataset: DataLoader, train: bool = True) -> Metrics:
        """
        This is an offline implementation only from IQLearn.
        Therefore, we are using the expert data as both policy and expert.
        """
        metrics = defaultdict(list)

        if train:
            self.policy.train()
            self.q_target.train()
        else:
            self.policy.eval()
            self.q_target.eval()

        pbar = dataset
        for batch in pbar:
            state, action, next_state, done = batch
            state = state.to(self.device)
            action = action.to(self.device)
            next_state = next_state.to(self.device)
            done = done.to(self.device)

            is_expert = torch.ones_like(done).to(self.device)
            expert_mask = is_expert.unsqueeze(-1).float()
            n_expert = expert_mask.sum().clamp(min=1)

            if train:
                self.optimizer_fn.zero_grad()

            # Compute Q(s, a) and V(s), V(s')
            current_q = self.policy(state)
            current_q = current_q.gather(1, action.reshape(-1, 1).long())

            current_v = self.get_value(state)

            # Compute target soft-value for next states (no gradients needed)
            with torch.no_grad():
                next_v = self.get_targetV(next_state)

            # Target: y = (1 - done) * γ * V(s')
            y = (1 - done.reshape(-1, 1).float()) * self.hyperparameters.get("gamma", 0.99) * next_v

            # Implied reward: r = Q(s, a) - y
            reward = current_q - y

            # ===== Component 1: Expert soft-Q loss =====
            phi_grad = reward * expert_mask
            soft_q_loss = -(phi_grad.sum() / n_expert)

            # ===== Component 2: Value function loss =====
            value_loss = ((current_v - y) * expert_mask).sum() / n_expert

            # ===== Component 3: Chi-squared regularization =====
            chi2_loss = torch.tensor(0.0).to(self.device)
            if self.hyperparameters.get("regularize", True):
                chi2_loss = 1.0 / (4.0 * self.hyperparameters.get("alpha", 0.5)) * (reward.pow(2)).mean()

            # ===== Total loss =====
            if self.hyperparameters.get("regularize", True):
                loss_q = soft_q_loss + value_loss + chi2_loss
            else:
                loss_q = soft_q_loss + value_loss

            if train:
                loss_q.backward()
                self.optimizer_fn.step()

            metrics["loss/q"].append(loss_q.item())
            metrics["loss/soft_q"].append(soft_q_loss.item())
            metrics["loss/value"].append(value_loss.item())
            metrics["loss/chi2"].append(chi2_loss.item())
            metrics["reward/mean"].append(reward.mean().item())
            metrics["value/policy"].append(current_v.mean().item())

        return {key: np.mean(value) for key, value in metrics.items()}
