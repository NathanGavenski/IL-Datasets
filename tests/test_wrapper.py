from unittest import TestCase

import gym
import gymnasium
import numpy as np
import pytest

from src.imitation_datasets.utils import GymWrapper, WrapperException

class TestWrapper(TestCase):

    def setUp(self) -> None:
        self.gym = gym.make("CartPole-v1")
        self.gymnasium = gymnasium.make("CartPole-v1", render_mode="rgb_array")

    def tearDown(self) -> None:
        self.gym.close()
        del self.gym

        self.gymnasium.close()
        del self.gymnasium

    def test_version_init(self) -> None:
        with pytest.raises(WrapperException):
            GymWrapper(self.gym, version="newest")

        with pytest.raises(WrapperException):
            GymWrapper(self.gymnasium, version="older")

        with pytest.raises(ValueError):
            GymWrapper(self.gym, version="oldest")

        with pytest.raises(ValueError):
            GymWrapper(self.gymnasium, version="newer")

    def test_action_space(self) -> None:
        assert GymWrapper(self.gymnasium).action_space == self.gymnasium.action_space
        assert GymWrapper(self.gym, version="older").action_space == self.gym.action_space

    def test_state_space(self) -> None:
        assert GymWrapper(self.gymnasium).observation_space == self.gymnasium.observation_space
        assert GymWrapper(self.gym, version="older").observation_space == self.gym.observation_space

    def test_reset(self) -> None:
        state = GymWrapper(self.gymnasium).reset()
        assert isinstance(state[0], np.float32)

        state = GymWrapper(self.gym, version="older").reset()
        assert isinstance(state[0], np.float32)

    def test_render(self) -> None:
        env = GymWrapper(self.gymnasium)
        env.reset()
        assert env.render().shape == (400, 600, 3)
        env.close()
        del env

        env = gymnasium.make("CartPole-v1")
        env = GymWrapper(env)
        env.reset()
        with pytest.raises(WrapperException):
            env.render()
        env.close()
        del env

        env = GymWrapper(self.gym, version="older")
        env.reset()
        assert env.render("rgb_array").shape == (400, 600, 3)

    def test_step(self) -> None:
        env = GymWrapper(self.gymnasium)
        env.reset()
        assert len(env.step(env.action_space.sample())) == 4

        env = GymWrapper(self.gym, version="older")
        env.reset()
        assert len(env.step(env.action_space.sample())) == 4
