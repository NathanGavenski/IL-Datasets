import shutil
import tempfile
from unittest import TestCase

import gymnasium as gym
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from src.imitation_datasets.baselines.methods import GAIL


def expert_loader(environment: gym.Env, discrete: bool) -> DataLoader:
    size = 128
    states = torch.rand(size, *environment.observation_space.shape)
    if discrete:
        actions = torch.randint(0, environment.action_space.n, (size, 1)).float()
    else:
        actions = torch.rand(size, *environment.action_space.shape)
    return DataLoader(TensorDataset(states, actions, states), batch_size=64, shuffle=True)


class TestGAIL(TestCase):

    def setUp(self) -> None:
        self.config = {"n_steps": 128, "batch_size": 32, "ppo_epochs": 2}
        # Checkpoints and logs go to a temporary folder, never to ./tmp or ./benchmark_results.
        self.folder = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_init_discrete(self) -> None:
        env = gym.make("CartPole-v1")
        method = GAIL(env)

        assert method.discrete
        assert method.policy.layers[0].in_features == 4
        assert method.discriminator.layers[0].in_features == 4 + 2
        assert method.log_std is None
        assert isinstance(method.optimizer_fn, optim.Adam)
        assert isinstance(method.loss_fn, nn.CrossEntropyLoss)
        assert method.save_path == "./tmp/gail/CartPole/"

    def test_init_continuous(self) -> None:
        env = gym.make("MountainCarContinuous-v0")
        method = GAIL(env)

        assert not method.discrete
        assert method.policy.layers[0].in_features == 2
        assert method.discriminator.layers[0].in_features == 2 + 1
        assert method.log_std.shape == (1,)
        assert len(method.optimizer_fn.param_groups) == 2
        assert method.save_path == "./tmp/gail/MountainCarContinuous/"

    def test_train_discrete(self) -> None:
        env = gym.make("CartPole-v1")
        method = GAIL(env, enjoy_criteria=1)
        method.save_path = f"{self.folder}/checkpoints/"
        method.hyperparameters.update(self.config)
        method.train(2, expert_loader(env, discrete=True), folder=f"{self.folder}/logs")

        action = method.predict(env.reset()[0])
        assert env.action_space.contains(int(action))

    def test_train_continuous(self) -> None:
        env = gym.make("MountainCarContinuous-v0")
        method = GAIL(env, enjoy_criteria=1)
        method.save_path = f"{self.folder}/checkpoints/"
        method.hyperparameters.update(self.config)
        log_std = method.log_std.detach().clone()
        method.train(2, expert_loader(env, discrete=False), folder=f"{self.folder}/logs")

        assert method.predict(env.reset()[0]).shape == (1,)
        assert not torch.equal(log_std, method.log_std.detach())

    def test_save_load(self) -> None:
        env = gym.make("CartPole-v1")
        method = GAIL(env)
        method.normalizer.update(torch.rand(10, 4))
        method.save(path=self.folder)

        loaded = GAIL(env).load(path=self.folder)
        for key, value in method.policy.state_dict().items():
            assert torch.equal(value, loaded.policy.state_dict()[key])
        assert torch.equal(method.normalizer.mean, loaded.normalizer.mean)
        assert torch.equal(method.normalizer.var, loaded.normalizer.var)
