"""Module for testing util functions"""
from unittest import TestCase

import pytest
import gym
import gymnasium
import numpy as np

from imitation_datasets import utils


class TestGymWrapper(TestCase):
    """Test Gym Wrapper functions"""

    def setUp(self) -> None:
        self.env = gym.make("CartPole-v1")
        self.env_new = gymnasium.make("CartPole-v1", render_mode="rgb_array")

    def tearDown(self) -> None:
        self.env.close()
        del self.env

        self.env_new.close()
        del self.env_new

    def test_init(self) -> None:
        env = utils.GymWrapper(self.env, version="older")
        assert isinstance(env, utils.GymWrapper)
        assert isinstance(env.env, gym.Env)

        env = utils.GymWrapper(self.env_new, version="newest")
        assert isinstance(env, utils.GymWrapper)
        assert isinstance(env.env, gymnasium.Env)

        with pytest.raises(ValueError):
            utils.GymWrapper(self.env, version="0.21.0")

        with pytest.raises(utils.WrapperException):
            utils.GymWrapper(self.env, version="newest")

        with pytest.raises(utils.WrapperException):
            utils.GymWrapper(self.env_new, version="older")

    def test_reset_21(self) -> None:
        env = utils.GymWrapper(self.env, version="older")
        state = env.reset()
        assert state.shape[0] == 4

    def test_reset_26(self) -> None:
        env = utils.GymWrapper(self.env_new, version="newest")
        state = env.reset()
        assert state.shape[0] == 4

    def test_step_21(self) -> None:
        env = utils.GymWrapper(self.env, version="older")
        env.reset()
        state, reward, done, info = env.step(env.action_space.sample())
        assert isinstance(state, np.ndarray)
        assert isinstance(reward, (float, int))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_26(self) -> None:
        env = utils.GymWrapper(self.env_new, version="newest")
        env.reset()
        state, reward, done, info = env.step(env.action_space.sample())
        assert isinstance(state, np.ndarray)
        assert isinstance(reward, (float, int))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_render_21(self) -> None:
        env = utils.GymWrapper(self.env, version="older")
        env.reset()
        state = env.render()
        assert isinstance(state, np.ndarray)

        state = env.render("human")
        assert isinstance(state, bool)

    def test_render_26(self) -> None:
        env = utils.GymWrapper(self.env_new, version="newest")
        env.reset()
        state = env.render()
        assert isinstance(state, np.ndarray)
