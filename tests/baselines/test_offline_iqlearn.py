import shutil
import tempfile
from unittest import TestCase

import gymnasium as gym
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.imitation_datasets.baselines.methods import OfflineIQLearn


def expert_loader(environment: gym.Env) -> DataLoader:
    size = 128
    states = torch.rand(size, *environment.observation_space.shape)
    actions = torch.randint(0, environment.action_space.n, (size, 1)).float()
    dones = torch.zeros(size, dtype=torch.bool)
    dones[-1] = True
    return DataLoader(TensorDataset(states, actions, states, dones), batch_size=64, shuffle=True)


class TestOfflineIQLearn(TestCase):

    def setUp(self) -> None:
        # Checkpoints and logs go to a temporary folder, never to ./tmp or ./benchmark_results.
        self.folder = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_init(self) -> None:
        env = gym.make("CartPole-v1")
        method = OfflineIQLearn(env)

        assert method.discrete
        assert method.policy.layers[-1].out_features == 2
        assert method.save_path == "./tmp/iqlearn/CartPole"

    def test_train_save_load(self) -> None:
        env = gym.make("CartPole-v1")
        method = OfflineIQLearn(env, enjoy_criteria=1)
        method.save_path = f"{self.folder}/checkpoints"
        method.train(2, expert_loader(env), folder=f"{self.folder}/logs")

        loaded = OfflineIQLearn(env).load(path=method.save_path)
        for key, value in method.policy.state_dict().items():
            assert torch.equal(value, loaded.policy.state_dict()[key])
        assert env.action_space.contains(int(loaded.predict(env.reset()[0])))
