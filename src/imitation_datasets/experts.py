"""Helper classes for loading and using expert policies."""
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Union, Dict

import gymnasium as gym
from huggingface_sb3 import load_from_hub
from stable_baselines3.common.base_class import BaseAlgorithm

from .register import atari, classic, mujoco


@dataclass
class Policy:
    """Policy dataclass to load and use expert policies."""

    name: str
    repo_id: str
    filename: str
    threshold: float
    algo: BaseAlgorithm
    policy: BaseAlgorithm = field(init=False, default=None)
    internal_state: Any = field(init=False, default=None)
    environment: Any = field(init=False, default=None)

    def load(self) -> BaseAlgorithm:
        """
        Load policy from HuggingFace hub.
        It uses a custom_object to replicate stable_baselines behaviour.

        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0
        }

        Returns:
            BaseAlgorithm: Stable baseline policy loaded from HuggingFace hub.
        """
        checkpoint = load_from_hub(
            repo_id=self.repo_id,
            filename=self.filename,
        )

        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0
        }

        self.policy = self.algo.load(
            checkpoint,
            custom_objects=custom_objects
        )
        return self.policy

    def predict(
            self,
            obs: List[Union[int, float]],
            deterministic: bool = True
    ) -> Tuple[
        Union[int, float, List[Union[int, float]]],
        Union[int, float, List[Union[int, float]]]
    ]:
        """
        Predict action given observation.

        Args:
            obs (List[int | float]): observation from environment.
            deterministic (bool, optional): Use exploration to predict action. Defaults to True.

        Returns:
            action (Union[int, float, List[Union[int, float]]]):
                action predicted by the policy.
            internal_states (Union[int, float, List[Union[int, float]]]):
                internal states of the policy.

        Note: typing depends on the environment.
        """
        action, internal_states = self.policy.predict(
            obs,
            state=self.internal_state,
            deterministic=deterministic,
        )
        self.internal_state = internal_states
        return action, internal_states

    def get_environment(self) -> str:
        """Return environment name.

        Returns:
            str: environment name.
        """
        if self.environment is None:
            self.environment = gym.make(self.name, render_mode="rgb_array")
        return self.environment


class Experts:
    """Helper class to register and get expert policies."""
    experts: Dict[str, Policy] = {
        key: Policy(**value) for env in [atari, classic, mujoco] for key, value in env.items()
    }

    @classmethod
    def register(cls, identifier: str, policy: Policy) -> None:
        """Register a new policy."""
        if not isinstance(policy.threshold, float):
            policy.threshold = float(policy.threshold)

        cls.experts[identifier] = policy

    @classmethod
    def get_expert(cls, identifier: str) -> Policy:
        """Return expert policy.

        Args:
            identifier (str): identifier of the policy.

        Returns:
            Policy: dataclass with expert policy information.
        """
        return cls.experts[identifier]

    @classmethod
    def get_register(cls) -> None:
        """Print entire register of expert policies."""
        print(cls.experts)
