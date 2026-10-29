"""Module for Generative Adversarial Imitation Learning"""
import os
from collections import defaultdict
from typing import Dict, Iterator, Tuple

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from gymnasium import Env
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical, Normal
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tensorboard_wrapper.tensorboard import Tensorboard
from tqdm import tqdm

from .method import Method, Metrics
from .policies import MLP
from .utils import import_hyperparameters


PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = f"{PATH}/config/gail.yaml"


class RunningNormalizer(nn.Module):
    """Running mean and standard deviation for observation normalization."""

    def __init__(self, size: int, clip: float = 10.0) -> None:
        """Initialize normalizer.

        Args:
            size (int): observation size.
            clip (float): value to clip normalized observations. Defaults to 10.
        """
        super().__init__()
        self.clip = clip
        self.register_buffer("mean", torch.zeros(size, dtype=torch.float64))
        self.register_buffer("var", torch.ones(size, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float64))

    def update(self, x: torch.Tensor) -> None:
        """Update statistics with a batch of observations (parallel algorithm).

        Args:
            x (torch.Tensor): observations (B, C).
        """
        x = x.double()
        batch_mean, batch_var, batch_count = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta.pow(2) * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize observations.

        Args:
            x (torch.Tensor): observations (B, C).

        Returns:
            x (torch.Tensor): normalized observations (B, C).
        """
        x = (x.double() - self.mean) / torch.sqrt(self.var + 1e-8)
        return x.clamp(-self.clip, self.clip).float()


class GAIL(Method):
    """Generative Adversarial Imitation Learning method based on (Ho and Ermon, 2016).

    The generator is optimized with PPO (Schulman et. al., 2017) instead of TRPO.
    Each training epoch collects `n_steps` environment steps, updates the
    discriminator against expert (state, action) pairs and then updates the
    policy with the discriminator reward.
    """

    __version__ = "1.0.0"
    __author__ = "Ho and Ermon"
    __method_name__ = "Generative Adversarial Imitation Learning"

    def __init__(
        self,
        environment: Env,
        enjoy_criteria: int = 100,
        verbose: bool = False,
        config_file: str = None
    ) -> None:
        """Initialize GAIL method."""
        self.enjoy_criteria = enjoy_criteria
        self.verbose = verbose
        try:
            self.environment_name = environment.spec.name
        except AttributeError:
            self.environment_name = environment.spec._env_name
        self.save_path = f"./tmp/gail/{self.environment_name}/"

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
        # Rollouts are collected one step at a time, so everything stays on CPU.
        self.device = "cpu"

        self.normalizer = RunningNormalizer(self.observation_size)
        self.critic = MLP(self.observation_size, 1)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.hyperparameters["critic_lr"]
        )

        self.discriminator = MLP(self.observation_size + self.action_size, 1)
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=self.hyperparameters["discriminator_lr"]
        )

        self.log_std = None
        if not self.discrete:
            self.log_std = nn.Parameter(torch.zeros(self.action_size))
            self.optimizer_fn.add_param_group({"params": [self.log_std]})
            self.action_low = torch.as_tensor(environment.action_space.low, dtype=torch.float32)
            self.action_high = torch.as_tensor(environment.action_space.high, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for the method.

        Args:
            x (torch.Tensor): input.

        Returns:
            x (torch.Tensor): logits (discrete) or action mean (continuous).
        """
        return self.policy(self.normalizer(x))

    def distribution(self, x: torch.Tensor) -> torch.distributions.Distribution:
        """Action distribution for the current policy.

        Args:
            x (torch.Tensor): input.

        Returns:
            distribution (Distribution): Categorical or diagonal Normal distribution.
        """
        output = self.forward(x)
        if self.discrete:
            return Categorical(logits=output)
        return Normal(output, self.log_std.exp().expand_as(output))

    def discriminator_logits(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Discriminator logits, where positive values mean expert.

        Args:
            state (torch.Tensor): states (B, C).
            action (torch.Tensor): action indexes (B,) or actions (B, A).

        Returns:
            logits (torch.Tensor): discriminator logits (B,).
        """
        if self.discrete:
            action = F.one_hot(action.long().view(-1), self.action_size).float()
        x = torch.cat([self.normalizer(state), action.float()], dim=1)
        return self.discriminator(x).squeeze(1)

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
                "normalizer": self.normalizer.state_dict(),
                "critic": self.critic.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "log_std": self.log_std,
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
        self.normalizer.load_state_dict(extra["normalizer"])
        self.critic.load_state_dict(extra["critic"])
        self.discriminator.load_state_dict(extra["discriminator"])
        if self.log_std is not None:
            self.log_std.data.copy_(extra["log_std"].data)

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
            n_epochs (int): amount of epoch to run, each epoch is `n_steps` environment steps.
            train_dataset (DataLoader): expert data to train the discriminator.
            eval_dataset (DataLoader): not used by GAIL. Defaults to None.
            always_save (bool): whether it should save all eval steps.
            folder (str): a specific folder to save the benchmark results.

        Returns:
            method (Self): trained method.
        """
        if folder is None:
            folder = f"./benchmark_results/gail/{self.environment_name}"
        if not os.path.exists(folder):
            os.makedirs(f"{folder}/")

        board = Tensorboard(path=folder)
        board.add_hparams(self.hyperparameters)
        self.policy.to(self.device)

        expert = self._expert_sampler(train_dataset)
        obs, _ = self.environment.reset()
        self._accumulated_reward = 0.0
        best_model = -np.inf

        pbar = range(n_epochs)
        if self.verbose:
            pbar = tqdm(pbar, desc=self.__method_name__)
        for epoch in pbar:
            rollout, obs, episode_rewards = self._rollout(obs)
            discriminator_metrics = self._train_discriminator(rollout, expert)
            policy_metrics = self._train(rollout)

            train_metrics = {**discriminator_metrics, **policy_metrics}
            if episode_rewards:
                train_metrics["rollout_reward"] = float(np.mean(episode_rewards))
            board.add_scalars("Train", epoch="train", **train_metrics)
            board.step("train")

            if epoch > 0 and epoch % self.enjoy_criteria == 0:
                metrics = self._enjoy()
                board.add_scalars("Enjoy", epoch="enjoy", **metrics)
                board.step("enjoy")
                # _enjoy shares the environment, so the rollout episode restarts.
                obs, _ = self.environment.reset()
                self._accumulated_reward = 0.0
                if best_model < metrics["aer"] or always_save:
                    best_model = metrics["aer"]
                    self.save(name=epoch if always_save else None)

                    if self.early_stop(metrics["aer"]):
                        return self

        return self

    def _expert_sampler(self, dataset: DataLoader) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Endless iterator over expert (state, action) batches.

        Args:
            dataset (DataLoader): expert data.

        Yields:
            state, action (Tuple[torch.Tensor, torch.Tensor]): expert batch.
        """
        while True:
            for state, action, *_ in dataset:
                if self.discrete:
                    action = action.view(-1)
                yield state.float(), action.float()

    @torch.no_grad()
    def _rollout(self, obs: np.ndarray) -> Tuple[Dict[str, torch.Tensor], np.ndarray, list]:
        """Collect `n_steps` environment steps with the current policy.

        Args:
            obs (np.ndarray): current observation.

        Returns:
            rollout (Dict[str, torch.Tensor]): states, actions, log probabilities,
                values, next values and episode ends.
            obs (np.ndarray): observation to continue from.
            episode_rewards (list): environment returns for finished episodes.
        """
        self.policy.eval()
        rollout = defaultdict(list)
        episode_rewards, accumulated_reward = [], self._accumulated_reward

        for _ in range(self.hyperparameters["n_steps"]):
            state = torch.as_tensor(obs, dtype=torch.float32)[None]
            self.normalizer.update(state)
            distribution = self.distribution(state)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)

            if self.discrete:
                env_action = action.item()
                stored_action = action.float()
            else:
                log_prob = log_prob.sum(-1)
                action = action[0]
                stored_action = torch.max(torch.min(action, self.action_high), self.action_low)
                env_action = stored_action.numpy()

            obs, reward, terminated, truncated, _ = self.environment.step(env_action)
            accumulated_reward += reward
            next_state = torch.as_tensor(obs, dtype=torch.float32)[None]

            rollout["state"].append(state[0])
            rollout["action"].append(action.view(-1) if not self.discrete else action.float())
            rollout["env_action"].append(stored_action.view(-1))
            rollout["log_prob"].append(log_prob.view(()))
            rollout["value"].append(self.critic(self.normalizer(state)).view(()))
            next_value = 0.0 if terminated else self.critic(self.normalizer(next_state)).item()
            rollout["next_value"].append(torch.tensor(next_value))
            rollout["end"].append(torch.tensor(float(terminated or truncated)))

            if terminated or truncated:
                episode_rewards.append(accumulated_reward)
                accumulated_reward = 0.0
                obs, _ = self.environment.reset()

        self._accumulated_reward = accumulated_reward
        rollout = {key: torch.stack(value) for key, value in rollout.items()}
        if self.discrete:
            rollout["action"] = rollout["action"].view(-1)
            rollout["env_action"] = rollout["env_action"].view(-1)
        return rollout, obs, episode_rewards

    def _train_discriminator(
        self,
        rollout: Dict[str, torch.Tensor],
        expert: Iterator[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Metrics:
        """Discriminator loop: expert pairs are labelled 1 and policy pairs 0.

        Args:
            rollout (Dict[str, torch.Tensor]): policy data.
            expert (Iterator): expert batches.

        Returns:
            metrics (Metrics): discriminator loss and accuracies.
        """
        self.discriminator.train()
        batch_size = self.hyperparameters["batch_size"]
        n_steps = rollout["state"].shape[0]
        accumulated_loss, expert_accuracy, policy_accuracy = [], [], []

        for _ in range(self.hyperparameters["discriminator_epochs"]):
            for indexes in torch.randperm(n_steps).split(batch_size):
                expert_state, expert_action = next(expert)
                expert_state = expert_state[:indexes.shape[0]]
                expert_action = expert_action[:indexes.shape[0]]

                expert_logits = self.discriminator_logits(expert_state, expert_action)
                policy_logits = self.discriminator_logits(
                    rollout["state"][indexes],
                    rollout["env_action"][indexes]
                )
                loss = F.binary_cross_entropy_with_logits(
                    expert_logits, torch.ones_like(expert_logits)
                ) + F.binary_cross_entropy_with_logits(
                    policy_logits, torch.zeros_like(policy_logits)
                )

                self.discriminator_optimizer.zero_grad()
                loss.backward()
                self.discriminator_optimizer.step()

                accumulated_loss.append(loss.item())
                expert_accuracy.append((expert_logits > 0).float().mean().item())
                policy_accuracy.append((policy_logits < 0).float().mean().item())

        return {
            "discriminator_loss": np.mean(accumulated_loss),
            "discriminator_expert_accuracy": np.mean(expert_accuracy),
            "discriminator_policy_accuracy": np.mean(policy_accuracy),
        }

    def _train(self, rollout: Dict[str, torch.Tensor]) -> Metrics:
        """PPO loop with the discriminator reward and GAE.

        Args:
            rollout (Dict[str, torch.Tensor]): policy data.

        Returns:
            metrics (Metrics): policy loss, value loss, entropy and reward.
        """
        with torch.no_grad():
            logits = self.discriminator_logits(rollout["state"], rollout["env_action"])
            if self.hyperparameters["reward"] == "logit":
                reward = logits
            else:
                reward = -F.logsigmoid(-logits)

            gamma, gae_lambda = self.hyperparameters["gamma"], self.hyperparameters["gae_lambda"]
            delta = reward + gamma * rollout["next_value"] - rollout["value"]
            advantage = torch.zeros_like(delta)
            last = 0.0
            for step in reversed(range(delta.shape[0])):
                last = delta[step] + gamma * gae_lambda * (1 - rollout["end"][step]) * last
                advantage[step] = last
            returns = advantage + rollout["value"]

        self.policy.train()
        self.critic.train()
        batch_size = self.hyperparameters["batch_size"]
        clip_range = self.hyperparameters["clip_range"]
        n_steps = rollout["state"].shape[0]
        accumulated = defaultdict(list)

        for _ in range(self.hyperparameters["ppo_epochs"]):
            for indexes in torch.randperm(n_steps).split(batch_size):
                state = rollout["state"][indexes]
                batch_advantage = advantage[indexes]
                batch_advantage = (batch_advantage - batch_advantage.mean()) / (batch_advantage.std() + 1e-8)

                distribution = self.distribution(state)
                log_prob = distribution.log_prob(rollout["action"][indexes])
                entropy = distribution.entropy()
                if not self.discrete:
                    log_prob, entropy = log_prob.sum(-1), entropy.sum(-1)

                ratio = (log_prob - rollout["log_prob"][indexes]).exp()
                policy_loss = -torch.min(
                    ratio * batch_advantage,
                    ratio.clamp(1 - clip_range, 1 + clip_range) * batch_advantage
                ).mean()
                entropy = entropy.mean()
                loss = policy_loss - self.hyperparameters["entropy_coef"] * entropy

                self.optimizer_fn.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer_fn.step()

                value = self.critic(self.normalizer(state)).squeeze(1)
                value_loss = F.mse_loss(value, returns[indexes])
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

                accumulated["policy_loss"].append(policy_loss.item())
                accumulated["value_loss"].append(value_loss.item())
                accumulated["entropy"].append(entropy.item())

        metrics = {key: np.mean(value) for key, value in accumulated.items()}
        metrics["discriminator_reward"] = reward.mean().item()
        return metrics

    def _eval(self, dataset: DataLoader) -> Metrics:
        """GAIL does not use an evaluation dataset."""
        return {}

    def early_stop(self, metric: Metrics) -> bool:
        """Function that tells the method if it should stop or not.

        Args:
            metric (Metrics): the metric to evaluate whether it should early stop.

        Returns:
            stop (bool): if it should stop or not.
        """
        return False
