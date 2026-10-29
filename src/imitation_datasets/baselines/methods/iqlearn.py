"""Module for (online) Inverse soft-Q Learning"""
from collections import defaultdict
import copy
import os
from typing import Dict, Iterator, Tuple

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import gymnasium as gym
from gymnasium import Env
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.utils.data import DataLoader
from tensorboard_wrapper.tensorboard import Tensorboard
from tqdm import tqdm

from imitation_datasets.dataset.metrics import average_episodic_reward
from .method import Method, Metrics
from .policies import MLP
from .utils import import_hyperparameters


PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = f"{PATH}/config/iqlearn.yaml"


class QCritic(nn.Module):
    """Q(s, a) network for continuous actions."""

    def __init__(self, in_dim: int, hidden_dim: list = None, activation: nn.Module = nn.ReLU) -> None:
        super().__init__()
        self.q = MLP(in_dim, 1, hidden_dim, activation=activation)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([obs.float(), action.float()], dim=-1))


class ReplayBuffer:
    """Fixed size buffer for the agent transitions."""

    def __init__(self, size: int, observation_size: int, action_size: int) -> None:
        self.size, self.index, self.full = size, 0, False
        self.states = torch.zeros(size, observation_size)
        self.next_states = torch.zeros(size, observation_size)
        self.actions = torch.zeros(size, action_size)
        self.dones = torch.zeros(size, 1)

    def __len__(self) -> int:
        return self.size if self.full else self.index

    def add(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, done: bool) -> None:
        self.states[self.index] = torch.as_tensor(state, dtype=torch.float32)
        self.actions[self.index] = torch.as_tensor(action, dtype=torch.float32)
        self.next_states[self.index] = torch.as_tensor(next_state, dtype=torch.float32)
        self.dones[self.index] = float(done)
        self.index = (self.index + 1) % self.size
        self.full = self.full or self.index == 0

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        indexes = torch.randint(0, len(self), (batch_size,))
        return self.states[indexes], self.actions[indexes], self.next_states[indexes], self.dones[indexes]


class IQLearn(Method):
    """Inverse soft-Q Learning method based on (Garg et. al., 2021) for continuous actions.

    Online version used for the MuJoCo experiments of the paper: a SAC agent interacts
    with the environment and, at every step, the critic is trained with the IQ loss on
    expert and agent transitions, followed by a SAC actor update. Each training epoch is
    `steps_per_epoch` environment steps. For discrete actions use OfflineIQLearn.
    """

    __version__ = "1.0.0"
    __author__ = "Garg et. al."
    __method_name__ = "Inverse soft-Q Learning"

    def __init__(
        self,
        environment: Env,
        enjoy_criteria: int = 100,
        verbose: bool = False,
        config_file: str = None
    ) -> None:
        """Initialize IQLearn method."""
        self.enjoy_criteria = enjoy_criteria
        self.verbose = verbose
        try:
            self.environment_name = environment.spec.name
        except AttributeError:
            self.environment_name = environment.spec._env_name
        self.save_path = f"./tmp/iqlearn_online/{self.environment_name}/"

        if config_file is None:
            config_file = CONFIG_FILE

        self.hyperparameters = import_hyperparameters(
            config_file,
            self.environment_name,
        )

        super().__init__(
            environment,
            self.hyperparameters,
            activation=nn.ReLU
        )
        if self.discrete:
            raise ValueError("IQLearn only supports continuous actions, use OfflineIQLearn instead.")
        # Rollouts are collected one step at a time, so everything stays on CPU.
        self.device = "cpu"

        # SAC actor: the policy outputs the mean and log std of a tanh-squashed Gaussian.
        hidden_dim = self.hyperparameters.get("hidden_dim", None)
        self.policy = MLP(self.observation_size, self.action_size * 2, hidden_dim, activation=nn.ReLU)
        self.optimizer_fn = optim.Adam(self.policy.parameters(), lr=self.hyperparameters["lr"])

        self.critic = QCritic(self.observation_size + self.action_size, hidden_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.hyperparameters["critic_lr"]
        )
        self.alpha = self.hyperparameters["alpha"]

        low = torch.as_tensor(environment.action_space.low, dtype=torch.float32)
        high = torch.as_tensor(environment.action_space.high, dtype=torch.float32)
        self.action_scale = (high - low) / 2
        self.action_bias = (high + low) / 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for the method.

        Args:
            x (torch.Tensor): input.

        Returns:
            x (torch.Tensor): deterministic action (tanh of the mean, scaled to the action space).
        """
        mean, _ = self.policy(x).chunk(2, dim=-1)
        return torch.tanh(mean) * self.action_scale + self.action_bias

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample an action with its log probability (tanh-squashed Gaussian).

        Args:
            obs (torch.Tensor): observations (B, C).

        Returns:
            action (torch.Tensor): actions scaled to the action space (B, A).
            log_prob (torch.Tensor): log probability of the actions (B, 1).
        """
        mean, log_std = self.policy(obs).chunk(2, dim=-1)
        log_std = -5 + 0.5 * (2 - (-5)) * (torch.tanh(log_std) + 1)
        normal = Normal(mean, log_std.exp())
        x = normal.rsample()
        y = torch.tanh(x)
        action = y * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x) - torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)

    def get_value(self, obs: torch.Tensor, critic: nn.Module = None) -> torch.Tensor:
        """Soft value V(s) = Q(s, a) - alpha * log pi(a|s), with a ~ pi(s).

        Args:
            obs (torch.Tensor): observations (B, C).
            critic (nn.Module): critic to use. Defaults to the online critic.

        Returns:
            value (torch.Tensor): soft values (B, 1).
        """
        critic = self.critic if critic is None else critic
        action, log_prob = self.sample(obs)
        return critic(obs, action) - self.alpha * log_prob

    def save(self, path: str = None, name: str = None) -> None:
        """Save all model weights.

        Args:
            path (str): where to save the models. Defaults to None.
            name (str): name for the policy checkpoint. Defaults to None.
        """
        path = self.save_path if path is None else path
        if not os.path.exists(path):
            os.makedirs(path)

        name = "best_model" if name is None else name

        torch.save(self.policy.state_dict(), f"{path}/{name}.ckpt")
        torch.save(
            {
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
            },
            f"{path}/{name}_extra.ckpt"
        )

    def load(self, path: str = None, name: str = None) -> Self:
        """Load all model weights.

        Args:
            path (str): where to look for the model's weights. Defaults to None.
            name (str): name for the policy checkpoint. Defaults to None.

        Raises:
            ValueError: if the path does not exist.
        """
        path = self.save_path if path is None else path
        name = "best_model" if name is None else name

        if not os.path.exists(path):
            raise ValueError("Path does not exists.")

        device = torch.device(self.device)
        self.policy.load_state_dict(torch.load(f"{path}/{name}.ckpt", map_location=device))
        extra = torch.load(f"{path}/{name}_extra.ckpt", map_location=device)
        self.critic.load_state_dict(extra["critic"])
        self.critic_target.load_state_dict(extra["critic_target"])

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
            n_epochs (int): amount of epoch to run, each epoch is `steps_per_epoch` environment steps.
            train_dataset (DataLoader): expert data, sampled uniformly at every update.
            eval_dataset (DataLoader): not used by IQLearn. Defaults to None.
            always_save (bool): whether it should save all eval steps.
            folder (str): a specific folder to save the benchmark results.

        Returns:
            method (Self): trained method.
        """
        if folder is None:
            folder = f"./benchmark_results/iqlearn_online/{self.environment_name}"
        if not os.path.exists(folder):
            os.makedirs(f"{folder}/")

        board = Tensorboard(path=folder)
        hparams = {}
        for key, value in self.hyperparameters.items():
            if isinstance(value, list):
                value = ",".join([str(v) for v in value])
            hparams[key] = value
        board.add_hparams(hparams)

        expert = self._expert_sampler(train_dataset.dataset)
        steps_per_epoch = int(self.hyperparameters["steps_per_epoch"])
        buffer = ReplayBuffer(
            min(int(self.hyperparameters["replay_mem"]), n_epochs * steps_per_epoch),
            self.observation_size,
            self.action_size
        )
        obs, _ = self.environment.reset()
        accumulated_reward = 0.0
        best_model = -np.inf

        pbar = range(n_epochs)
        if self.verbose:
            pbar = tqdm(pbar, desc=self.__method_name__)
        for epoch in pbar:
            metrics, episode_rewards = defaultdict(list), []
            for _ in range(steps_per_epoch):
                with torch.no_grad():
                    action, _ = self.sample(torch.as_tensor(obs, dtype=torch.float32)[None])
                action = action[0].numpy()
                next_obs, reward, terminated, truncated, _ = self.environment.step(action)
                # Time limits are not terminal, so their value is still bootstrapped.
                buffer.add(obs, action, next_obs, terminated)
                accumulated_reward += reward
                obs = next_obs
                if terminated or truncated:
                    episode_rewards.append(accumulated_reward)
                    accumulated_reward = 0.0
                    obs, _ = self.environment.reset()

                if len(buffer) > self.hyperparameters["initial_mem"]:
                    for key, value in self._update(buffer, expert).items():
                        metrics[key].append(value)

            train_metrics = {key: float(np.mean(value)) for key, value in metrics.items()}
            if episode_rewards:
                train_metrics["rollout_reward"] = float(np.mean(episode_rewards))
            if train_metrics:
                board.add_scalars("Train", epoch="train", **train_metrics)
            board.step("train")

            if epoch > 0 and epoch % self.enjoy_criteria == 0:
                # As the official implementation, the best checkpoint is selected with a few
                # evaluation episodes (eval_episodes) on a separate environment.
                enjoy_metrics = self._evaluate(int(self.hyperparameters.get("eval_episodes", 10)))
                board.add_scalars("Enjoy", epoch="enjoy", **enjoy_metrics)
                board.step("enjoy")
                if best_model < enjoy_metrics["aer"] or always_save:
                    best_model = enjoy_metrics["aer"]
                    self.save(name=epoch if always_save else None)

                    if self.early_stop(enjoy_metrics["aer"]):
                        return self

        return self

    @torch.no_grad()
    def _evaluate(self, n_episodes: int) -> Metrics:
        """Evaluate the deterministic policy on a separate environment.

        Args:
            n_episodes (int): number of episodes.

        Returns:
            metrics (Metrics): average episodic reward and its standard deviation.
        """
        environment = gym.make(self.environment.spec.id)
        rewards = []
        for _ in range(n_episodes):
            obs, _ = environment.reset()
            done, accumulated_reward = False, 0.0
            while not done:
                obs, reward, terminated, truncated, _ = environment.step(self.predict(obs))
                accumulated_reward += reward
                done = terminated or truncated
            rewards.append(accumulated_reward)
        environment.close()
        return average_episodic_reward(rewards)

    def _expert_sampler(self, dataset) -> Iterator[Tuple[torch.Tensor, ...]]:
        """Endless iterator over uniformly sampled expert batches.

        Args:
            dataset (Dataset): expert dataset indexable by a tensor of indexes.

        Yields:
            state, action, next_state, done (Tuple[torch.Tensor, ...]): expert batch.
        """
        batch_size = self.hyperparameters["batch_size"]
        while True:
            state, action, next_state, done, *_ = dataset[torch.randint(0, len(dataset), (batch_size,))]
            yield (
                state.float(),
                action.float().reshape(-1, self.action_size),
                next_state.float(),
                done.float().reshape(-1, 1)
            )

    def _update(self, buffer: ReplayBuffer, expert: Iterator[Tuple[torch.Tensor, ...]]) -> Dict[str, float]:
        """One IQ critic update, one SAC actor update and a soft target update.

        Args:
            buffer (ReplayBuffer): agent transitions.
            expert (Iterator): expert batches.

        Returns:
            metrics (Dict[str, float]): losses for this update.
        """
        self.policy.train()
        self.critic.train()
        gamma = self.hyperparameters["gamma"]

        policy_batch = buffer.sample(self.hyperparameters["batch_size"])
        expert_batch = next(expert)
        state, action, next_state, done = [
            torch.cat([policy, demo], dim=0) for policy, demo in zip(policy_batch, expert_batch)
        ]
        is_expert = torch.cat([
            torch.zeros(policy_batch[0].shape[0], dtype=torch.bool),
            torch.ones(expert_batch[0].shape[0], dtype=torch.bool),
        ])

        # ===== Critic: IQ loss =====
        current_q = self.critic(state, action)
        current_v = self.get_value(state)
        with torch.no_grad():
            next_v = self.get_value(next_state, self.critic_target)
        y = (1 - done) * gamma * next_v

        # -E_expert[Q(s, a) - gamma V(s')]
        reward = current_q - y
        loss = -reward[is_expert].mean()
        metrics = {"loss/soft_q": loss.item()}

        if self.hyperparameters["loss"] == "v0":
            # (1 - gamma) E_expert[V(s0)], expert states as initial states
            value_loss = (1 - gamma) * current_v[is_expert].mean()
        else:
            # E_(expert and agent)[V(s) - gamma V(s')]
            value_loss = (current_v - y).mean()
        loss = loss + value_loss
        metrics["loss/value"] = value_loss.item()

        if self.hyperparameters.get("regularize", True):
            # chi-squared divergence on expert and agent transitions
            chi2_loss = 1 / (4 * self.hyperparameters["chi2_alpha"]) * reward.pow(2).mean()
            loss = loss + chi2_loss
            metrics["loss/chi2"] = chi2_loss.item()

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        # ===== Actor: SAC update on expert and agent states =====
        policy_action, log_prob = self.sample(state)
        actor_loss = (self.alpha * log_prob - self.critic(state, policy_action)).mean()
        self.optimizer_fn.zero_grad()
        actor_loss.backward()
        self.optimizer_fn.step()

        tau = self.hyperparameters["tau"]
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        metrics.update({
            "loss/critic": loss.item(),
            "loss/actor": actor_loss.item(),
            "entropy": -log_prob.mean().item(),
            "reward/expert": reward[is_expert].mean().item(),
            "reward/policy": reward[~is_expert].mean().item(),
        })
        return metrics

    def _train(self, dataset: DataLoader) -> Metrics:
        """IQLearn trains inside `train` because it interacts with the environment."""
        raise NotImplementedError()

    def _eval(self, dataset: DataLoader) -> Metrics:
        """IQLearn does not use an evaluation dataset."""
        return {}

    def early_stop(self, metric: Metrics) -> bool:
        """Function that tells the method if it should stop or not.

        Args:
            metric (Metrics): the metric to evaluate whether it should early stop.

        Returns:
            stop (bool): if it should stop or not.
        """
        return False
