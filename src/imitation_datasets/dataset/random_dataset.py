"""Module for creaiting a random dataset."""
from argparse import ArgumentParser, Namespace
import os
import shutil
from typing import List, Dict

import numpy as np
from tqdm import tqdm

from imitation_datasets import Controller
from imitation_datasets.experts import Policy, Experts
from imitation_datasets.utils import Context, GymWrapper
from imitation_datasets.dataset import BaselineDataset


def enjoy(expert: Policy, path: str, context: Context) -> bool:
    """Random enjoy function. It has an expert as parameter, but is only for
    using the Controller and multithreading."""
    done = False
    env = GymWrapper(expert.get_environment(), version="newest")

    states = []
    actions = []
    rewards = []
    state = env.reset()
    acc_reward = 0

    while not done:
        action = env.action_space.sample()
        states.append(state)
        actions.append(action)

        state, reward, done, _ = env.step(action)
        acc_reward += reward
        rewards.append(reward)
    env.close()

    episode_returns = np.array([acc_reward])

    episode = {
        'obs': np.array(states),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'episode_returns': episode_returns
    }
    if acc_reward >= expert.threshold:
        np.savez(f'{path}{context.index}', **episode)
        context.add_log(f'Accumulated reward {acc_reward}')
    return True


def collate(path: str, data: List[str]) -> bool:
    """Collate that outputs the same as StableBaselines."""
    episode = np.load(f'{path}{data[0]}')
    observation_space = episode["obs"].shape[1]

    states = np.ndarray(shape=(0, observation_space))
    episodes_starts = []
    actions = []
    rewards = []
    episode_returns = []

    for file in tqdm(data, desc="Collating random dataset"):
        episode = np.load(f'{path}{file}')
        states = np.append(states, episode['obs'], axis=0)
        actions += episode['actions'].tolist()
        rewards += episode['rewards'].tolist()
        episode_returns += episode['episode_returns'].tolist()

        episode_starts = np.zeros(episode['actions'].shape)
        episode_starts[0] = 1
        episodes_starts += episode_starts.tolist()

    states = states.reshape((-1, states.shape[-1]))

    actions = np.array(actions).reshape(-1)
    episodes_starts = np.array(episodes_starts).reshape(-1)

    rewards = np.array(rewards).reshape(-1)

    episode_returns = np.array(episode_returns).squeeze()

    episode = {
        'obs': states,
        'actions': actions,
        'rewards': rewards,
        'episode_returns': episode_returns,
        'episode_starts': episodes_starts
    }
    np.savez(f'{path}teacher', **episode)

    for file in data:
        os.remove(f'{path}{file}')

    return True


def create_arguments(args: Dict[str, str]) -> Namespace:
    """Recreate IL-Datasets arguments.

    Args:
        args (Dict[str, str]): arguments for creating random dataset.

    Returns:
        arguments (Namespace): parsed arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--game", type=str)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--mode", type=str)
    return parser.parse_args([item for pair in args.items() for item in pair])


def create_dataset(
    environment_name: str,
    episodes: int = 10000,
    threads: int = 4
) -> str:
    """Create random dataset.

    Args:
        environment_name (str): Gym id.
        episodes (int): how many episodes. Defaults to 10,000.
        threads (int): how many threads to use. Defaults to 4.

    Returns:
        path (str): path to dataset.
    """
    folder = f"random_{environment_name}"
    path = f"./dataset/{folder}"
    Experts.register(
        folder,
        Policy(
            name=environment_name,
            repo_id=None,
            filename=None,
            threshold=-np.inf,
            algo=None
        )
    )

    if os.path.exists(path):
        shutil.rmtree(path)

    args = create_arguments({
        "--game": folder,
        "--episodes": str(episodes),
        "--threads": str(threads),
        "--mode": "all"
    })

    controller = Controller(enjoy, collate,  args.episodes, args.threads)
    controller.start(args)
    return f"{path}/teacher.npz"


def get_random_dataset(
    environment_name: str,
    episodes: int = 10000,
    threads: int = 4
) -> BaselineDataset:
    """Create random dataset.

    Args:
        environment_name (str): Gym id.
        episodes (int): how many episodes. Defaults to 10,000.
        threads (int): how many threads to use. Defaults to 4.

    Returns:
        dataset (BaselineDataset): BaselineDataset object.
    """
    random_dataset = create_dataset(environment_name, episodes, threads)
    return BaselineDataset(random_dataset)
