from dataclasses import dataclass, field
from typing import Any, List, Tuple

from huggingface_sb3 import load_from_hub
from stable_baselines3.common.base_class import BaseAlgorithm

from .register import atari, classic, mujoco


@dataclass
class Policy:
    name: str
    repo_id: str
    filename: str
    threshold: float
    algo: BaseAlgorithm
    policy: BaseAlgorithm = field(init=False, default=None)
    internal_state: Any = field(init=False, default=None)

    def load(self) -> None:
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

    def predict(self, obs, deterministic: bool = True) -> Tuple[Any, Any]:
            action, internal_states = self.policy.predict(
                obs,
                state=self.internal_state,
                deterministic=deterministic,
            )
            self.internal_state = internal_states
            return action, internal_states

    def get_environment(self) -> str:
        return self.name


class Experts:

    experts: List[Policy] = {key: Policy(**value) for env in [atari, classic, mujoco] for key, value in env.items()}

    @classmethod
    def register(cls, identifier: str, policy: Policy) -> None:
        if not isinstance(policy.threshold, float):
            policy.threshold = float(policy.threshold)
        
        cls.experts[identifier] = policy

    @classmethod
    def get_expert(cls, identifier: str) -> Policy:
        return cls.experts[identifier]

    @classmethod
    def get_register(cls) -> str:
        print(cls.experts)
