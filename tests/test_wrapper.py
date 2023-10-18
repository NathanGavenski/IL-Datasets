"""Module for testing GymWrapper"""
from unittest import TestCase

import gym
import gymnasium
import numpy as np
import pytest

from src.imitation_datasets.utils import GymWrapper, WrapperException

class TestWrapper(TestCase):
    """Test cases for GymWrapper"""

    def setUp(self) -> None:
        """Setup gym and gymnasium environments."""
        self.gym = gym.make("CartPole-v1")
        self.gymnasium = gymnasium.make("CartPole-v1", render_mode="rgb_array")

    def tearDown(self) -> None:
        """Make sure environments are close and deleted."""
        self.gym.close()
        del self.gym

        self.gymnasium.close()
        del self.gymnasium

    def test_version_init(self) -> None:
        """Test different initialisations."""
        with pytest.raises(WrapperException):
            GymWrapper(self.gym, version="newest")

        with pytest.raises(WrapperException):
            GymWrapper(self.gymnasium, version="older")

        with pytest.raises(ValueError):
            GymWrapper(self.gym, version="oldest")

        with pytest.raises(ValueError):
            GymWrapper(self.gymnasium, version="newer")

    def test_action_space(self) -> None:
        """Test if action_space is consistent."""
        assert GymWrapper(self.gymnasium).action_space == self.gymnasium.action_space
        assert GymWrapper(self.gym, version="older").action_space == self.gym.action_space

    def test_state_space(self) -> None:
        """Test if observation_space is consistent."""
        assert GymWrapper(self.gymnasium).observation_space == self.gymnasium.observation_space
        assert GymWrapper(self.gym, version="older").observation_space == self.gym.observation_space

    def test_reset(self) -> None:
        """Test reseting gymnasium and gym."""
        state = GymWrapper(self.gymnasium).reset()
        assert isinstance(state[0], np.float32)

        state = GymWrapper(self.gym, version="older").reset()
        assert isinstance(state[0], np.float32)

    def test_render(self) -> None:
        """Test renderng gymnasium and gym."""
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
        """Test step function for gymnaiusm."""
        env = GymWrapper(self.gymnasium)
        env.reset()
        assert len(env.step(env.action_space.sample())) == 4

        env = GymWrapper(self.gym, version="older")
        env.reset()
        assert len(env.step(env.action_space.sample())) == 4
