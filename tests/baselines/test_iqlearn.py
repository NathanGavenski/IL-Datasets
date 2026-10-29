import shutil
import tempfile
from unittest import TestCase

import gymnasium as gym
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.imitation_datasets.baselines.methods import IQLearn


def expert_loader(environment: gym.Env) -> DataLoader:
    size = 128
    states = torch.rand(size, *environment.observation_space.shape)
    actions = torch.rand(size, *environment.action_space.shape) * 2 - 1
    dones = torch.zeros(size, dtype=torch.bool)
    dones[-1] = True
    return DataLoader(TensorDataset(states, actions, states, dones), batch_size=64, shuffle=True)


class TestIQLearn(TestCase):

    def setUp(self) -> None:
        self.config = {"steps_per_epoch": 64, "initial_mem": 16, "batch_size": 16}
        # Checkpoints and logs go to a temporary folder, never to ./tmp or ./benchmark_results.
        self.folder = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_init(self) -> None:
        env = gym.make("MountainCarContinuous-v0")
        method = IQLearn(env)

        assert not method.discrete
        assert method.policy.layers[-1].out_features == 2
        assert method.critic.q.layers[0].in_features == 2 + 1
        assert method.save_path == "./tmp/iqlearn_online/MountainCarContinuous/"

    def test_discrete_raises(self) -> None:
        with pytest.raises(ValueError):
            IQLearn(gym.make("CartPole-v1"))

    def test_train_save_load(self) -> None:
        env = gym.make("MountainCarContinuous-v0")
        method = IQLearn(env, enjoy_criteria=1)
        method.save_path = f"{self.folder}/checkpoints/"
        method.hyperparameters.update(self.config)
        target = [param.clone() for param in method.critic_target.parameters()]
        method.train(2, expert_loader(env), folder=f"{self.folder}/logs")

        assert any(
            not torch.equal(before, after)
            for before, after in zip(target, method.critic_target.parameters())
        )
        action = method.predict(env.reset()[0])
        assert action.shape == (1,)
        assert env.action_space.contains(action.astype(env.action_space.dtype))

        loaded = IQLearn(env).load(path=method.save_path)
        for key, value in method.critic.state_dict().items():
            assert torch.equal(value, loaded.critic.state_dict()[key])
