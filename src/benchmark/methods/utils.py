"""Module for utility functions

    load_hyperparameters: load hyperparameters from YAML file
    convert_hyperparameters: convert string values into other typings
    import_hyperparameters: loads and converts string values.
"""
from collections import deque
from typing import Dict, Any, Tuple, List
import os

import numpy as np
import torch
import yaml

from imitation_datasets.dataset import BaselineDataset


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int) -> None:
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=self.buffer_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self) -> List[torch.Tensor]:
        if self.batch_size > self.size():
            raise ValueError("Replay buffer doesn't have enough experiences")

        indexes = np.random.choice(np.arange(self.size()), size=self.batch_size, replace=False)
        agent = [self.buffer[index] for index in indexes]
        agent = [torch.from_numpy(np.array(x)) for x in zip(*agent)]

        indexes = np.random.choice(
            np.arange(len(self.dataset)),
            size=self.batch_size,
            replace=False
        )
        expert = self.dataset[indexes]
        return agent, expert

    def clear(self) -> None:
        self.buffer.clear()

    def save(self, path) -> None:
        buffer = np.asarray(self.buffer)
        np.save(path, buffer)

    def load(self, dataset: BaselineDataset) -> None:
        self.dataset = dataset
        self.dataset.get_reward_and_done = True

def load_hyperparameters(path: str, environment_name: str = None) -> Dict[str, Any]:
    """Load hyperparameters from YAML file.

    Args:
        path (str): path to the YAML file.
        environment_name (str): Entry name in YAML file. Defaults to None.

    Raises:
        FileNotFoundError: if YAML file does not exists.
        KeyError: if environment_name and 'Default' not in YAML file.

    Returns:
        hyperparameters (Dict[str, Any]): loaded hyperparameters.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exists")

    if environment_name is None:
        environment_name = 'Default'

    with open(path, 'r', encoding='utf-8') as _file:
        config = yaml.safe_load(_file)

    if environment_name not in config.keys() and 'Default' not in config.keys():
        raise KeyError(f"Default configuration missing from file at {path}")

    return config.get(environment_name, config['Default'])


def convert_hyperparameters(hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
    """Convert hyperparameters to correct typing.

    Args:
        hyperparameters (Dict[str, Any]): Dictionary with hyperparameters.

    Returns:
        hyperparameters (Dict[str, Any]): Dictionary with converted values.
    """
    for key, value in hyperparameters.items():
        if 'lr' in key:
            hyperparameters[key] = float(value)

        if 'random_episodes' == key:
            hyperparameters[key] = int(value)
    return hyperparameters


def import_hyperparameters(path: str, environment_name: str = None) -> Dict[str, Any]:
    """Import hyperparameters from YAML file.

    Args:
        path (str): path to the YAML file.
        environment_name (str): Entry name in YAML file. Defaults to None.

    Returns:
        hyperparameters (Dict[str, Any]): loaded hyperparameters.
    """
    return convert_hyperparameters(load_hyperparameters(path, environment_name))


def reached_goal(environment: str, gym_return: Tuple[Any], acc_reward: int) -> bool:
    """Check whether the agent has reached the goal of an environment.

    Args:
        environment (str): name of the environment.
        gym_return (Tuple[Any]): the gym return for one state.
        acc_reward (int): accumulated reward.

    Returns:
        reached_goal (bool): True if the agent has reached the goal.
    """
    obs, reward, *_ = gym_return
    environment_name = environment.lower()

    if "cartpole" in environment_name:
        # It could also be 195 if using CartPole-v0
        return acc_reward >= 500
    if "mountaincar" in environment_name:
        # Flag position
        return obs[0] >= 0.5
    if "acrobot" in environment_name:
        # So we don't have to have the environmnet
        return reward == 0

    raise ValueError("Environment has no goal registered")
