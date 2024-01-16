"""Module for registering environments and methods for benchmarking

Environments: each environment should be in one Dict[str, Dict[str, Any], where
    the first key should be the gym environment name and inside a path for the
    HuggingFace dataset and the reward for a random agent.

    To retrieve the reward of a random agent you can use the following code:
    ```
    rewards = []
    for _ in range(100):
        done = False
        env.reset()
        total_reward = 0
        while not done:
            _, _, reward, *_ = env.step(env.action_space.sample())
            total_reward += reward
        rewards.append(total_reward)
    np.mean(rewards)
    ```
"""
from ast import literal_eval
from typing import Any, Dict, List

from .methods import BC, BCO, ABCO, IUPE
from .methods.method import Method


classic_control: Dict[str, Dict[str, Any]] = {
    "CartPole-v1": {
        "path": "NathanGavenski/CartPole-v1",
        "random_reward": 9.8
    },
    "MountainCar-v0": {
        "path": "NathanGavenski/MountainCar-v0",
        "random_reward": -200
    },
    "Acrobot-v1": {
        "path": "NathanGavenski/Acrobot-v1",
        "random_reward": -498.65
    }
}


benchmark_environments: List[Dict[str, Dict[str, Any]]] = [
    classic_control
]


benchmark_methods: List[Method] = [
    BC,
    BCO,
    ABCO,
    IUPE
]


def get_methods(names: List[str]) -> List[Method]:
    """Get methods from string list.

    Args:
        names (List[str]): list of method names.

    Returns:
        benchmark_methods (List[Method]): list of methods.
    """
    if len(names) == 1 and names[0] == "all":
        return benchmark_methods

    partial_benchmark_methods = []
    for name in names:
        partial_benchmark_methods.append(eval(name.upper()))
    return partial_benchmark_methods
