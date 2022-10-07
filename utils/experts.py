from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, DefaultDict, List, Tuple

from huggingface_sb3 import load_from_hub
from stable_baselines3.common.base_class import BaseAlgorithm

from .register import atari, mujoco


@dataclass
class Policy:
    name: str
    repo_id: str
    filename: str
    threshold: float
    algo: BaseAlgorithm
    policy: BaseAlgorithm = field(init=False, default=None)

    def load(self) -> BaseAlgorithm:
        checkpoint = load_from_hub(
            repo_id=self.repo_id,
            filename=self.filename,
        )

        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0
        }
        
        return self.algo.load(
            checkpoint, 
            custom_objects=custom_objects
        )

    def predict(self, obs, state, deterministic: bool = True) -> Tuple[Any, Any]:
        raise NotImplementedError('TODO')


class Experts:

    experts: List[Policy] = {key: Policy(**value) for env in [atari, mujoco] for key, value in env.items()}

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
