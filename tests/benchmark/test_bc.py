from unittest import TestCase

import gymnasium as gym
from torch import nn, optim

from src.benchmark.methods.bc import BC


class TestBC(TestCase):

    def test_init_discrete(self) -> None:
        env = gym.make("CartPole-v1")
        method = BC(env)

        assert method.discrete
        assert method.action_size == env.action_space.n
        assert method.observation_size == env.observation_space.shape[0]
        assert method.policy.layers[0].in_features == 4
        assert isinstance(method.optimizer_fn, optim.Adam)
        assert isinstance(method.loss_fn, nn.CrossEntropyLoss)

    def test_init_continuous(self) -> None:
        env = gym.make("MountainCarContinuous-v0")
        method = BC(env)

        assert not method.discrete
        assert method.action_size == env.action_space.shape[0]
        assert method.observation_size == env.observation_space.shape[0]
        assert method.policy.layers[0].in_features == 2
        assert isinstance(method.optimizer_fn, optim.Adam)
        assert isinstance(method.loss_fn, nn.MSELoss)
