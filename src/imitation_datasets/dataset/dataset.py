"""Module for datasets"""
import os
from typing import Tuple

from datasets import load_dataset
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .huggingface import huggingface_to_baseline


class BaselineDataset(Dataset):
    """Teacher dataset for IL methods."""

    def __init__(self, path: str, source: str = "local", n_episodes: int = None) -> None:
        """Initialize dataset.

        Args:
            path (Str): path to the dataset.
            source (str): whether is a HuggingFace or a local dataset.
                Defaults to 'local'.

        Raises:
            ValueError: if path does not exist.
        """
        if source == "local" and not os.path.exists(path):
            raise ValueError(f"No dataset at: {path}")

        if source == "local":
            self.data = np.load(path, allow_pickle=True)
            self.average_reward = np.mean(self.data["episode_returns"])
        else:
            dataset = load_dataset(path, split="train")
            self.data = huggingface_to_baseline(dataset)
            self.average_reward = []

        self.states = np.ndarray(shape=(0, self.data["obs"].shape[-1]))
        self.next_states = np.ndarray(shape=(0, self.data["obs"].shape[-1]))

        if len(self.data["actions"].shape) == 1:
            action_size = 1
        else:
            action_size = self.data["actions"].shape[-1]
        self.actions = np.ndarray(shape=(0, action_size))

        episode_starts = np.where(self.data["episode_starts"] == 1)[0]
        episode_ends = [*episode_starts[1:], len(self.data["obs"])]

        if n_episodes is not None:
            episode_starts = episode_starts[:n_episodes]
            episode_ends = episode_ends[:n_episodes]

        for start, end in zip(tqdm(episode_starts), episode_ends):
            episode = self.data["obs"][start:end]
            actions = self.data["actions"][start:end - 1].reshape((-1, 1))
            self.actions = np.append(self.actions, actions, axis=0)
            self.states = np.append(self.states, episode[:-1], axis=0)
            self.next_states = np.append(self.next_states, episode[1:], axis=0)

            if source != "local":
                self.average_reward.append(self.data["rewards"][start:end].sum())

        if isinstance(self.average_reward, list):
            self.average_reward = np.mean(self.average_reward)

        assert self.states.shape[0] == self.actions.shape[0] == self.next_states.shape[0]

    def __len__(self) -> int:
        """Dataset length.

        Returns:
            length (int): length.
        """
        return self.states.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        """Get item from dataset.

        Args:
            index (int): index.

        Returns:
            state (torch.Tensor): state for timestep t.
            action (torch.Tensor): action for timestep t.
            next_state (torch.Tensor): state for timestep t + 1.
        """
        state = torch.from_numpy(self.states[index])
        action = torch.tensor([self.data["actions"][index]])
        next_state = torch.from_numpy(self.next_states[index])
        return state, action, next_state
