"""Module providing functions for Controllers"""
import os
from typing import List

import numpy as np

from .experts import Policy
from .utils import Context
from .utils import GymWrapper


def enjoy(expert: Policy, path: str, context: Context) -> bool:
    """
    This is a simple enjoy function example.
    It has three arguments and should return a boolean.
    """
    done = False
    expert.load()

    env = GymWrapper(expert.get_environment(), version="newest")

    states, actions = [], []
    acc_reward, state = 0, env.reset()
    while not done:
        action, _ = expert.predict(state)
        state, reward, done, _ = env.step(action)
        acc_reward += reward
        states.append(state)
        actions.append(action)
    env.close()

    episode = {
        'states': np.array(states),
        'actions': np.array(actions)
    }
    if acc_reward >= expert.threshold:
        np.savez(f'{path}{context.index}', **episode)
        context.add_log(f'Accumulated reward {acc_reward}')
    return acc_reward >= expert.threshold


def collate(path, data) -> bool:
    """This function is a simple collate function."""
    episodes_starts = []
    states, actions = [], []

    for file in data:
        episode = np.load(f'{path}{file}')
        states.append(episode['states'])
        actions.append(episode['actions'])

        episode_starts = np.zeros(episode['actions'].shape)
        episode_starts[0] = 1
        episodes_starts.append(episode_starts)

    states = np.array(states)
    states = states.reshape((-1, states.shape[-1]))
    actions = np.array(actions).reshape(-1)
    episodes_starts = np.array(episodes_starts).reshape(-1)

    episode = {
        'states': states,
        'actions': actions,
        'episode_starts': episodes_starts
    }
    np.savez(f'{path}teacher', **episode)

    for file in data:
        os.remove(f'{path}{file}')

    return True


def baseline_enjoy(expert: Policy, path: str, context: Context) -> bool:
    """Enjoy following StableBaseline output."""
    done = False
    expert.load()

    env = GymWrapper(expert.get_environment(), version="newest")

    states = []
    actions = []
    rewards = []
    state = env.reset()
    acc_reward = 0

    while not done:
        action, _ = expert.predict(state)
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
    return acc_reward >= expert.threshold


def baseline_collate(path: str, data: List[str]) -> bool:
    """Collate that outputs the same as StableBaseline."""
    episode = np.load(f'{path}{data[0]}')
    observation_space = episode["obs"].shape[1]

    states = np.ndarray(shape=(0, observation_space))
    episodes_starts = []
    actions = []
    rewards = []
    episode_returns = []

    for file in data:
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
