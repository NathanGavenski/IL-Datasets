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
from typing import Any, Dict

from methods.bc import BC


classic_control: Dict[str, Dict[str, Any]] = {
    "CartPole-v1": {
        "path": "NathanGavenski/CartPole-v1",
        "random_reward": 9.8
    }
}


benchmark_environments = [
    classic_control
]


benchmark_methods = [
    BC
]
