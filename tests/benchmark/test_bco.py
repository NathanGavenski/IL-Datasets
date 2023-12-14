from unittest import TestCase

import gymnasium as gym
from torch import nn, optim

from src.benchmark.methods import BCO


class TestBCO(TestCase):

    def test_init_discrete(self) -> None:
        env = gym.make("CartPole-v1")
        method = BCO(env)

        assert method.discrete
        assert method.policy.layers[0].in_features == 4
        assert isinstance(method.optimizer_fn, optim.Adam)
        assert isinstance(method.loss_fn, nn.CrossEntropyLoss)
        assert method.save_path == "./tmp/bco/CartPole/"

    def test_init_continuous(self) -> None:
        env = gym.make("MountainCarContinuous-v0")
        method = BCO(env)

        assert not method.discrete
        assert method.policy.layers[0].in_features == 2
        assert isinstance(method.optimizer_fn, optim.Adam)
        assert isinstance(method.loss_fn, nn.MSELoss)
        assert method.save_path == "./tmp/bco/MountainCarContinuous/"
