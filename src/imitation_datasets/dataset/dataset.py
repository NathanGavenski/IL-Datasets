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

    def __init__(
        self,
        path: str,
        source: str = "local",
        split: str = "train",
        n_episodes: int = None
    ) -> None:
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

        episode_starts = list(np.where(self.data["episode_starts"] == 1)[0])
        episode_starts.append(len(self.data["episode_starts"]))

        if n_episodes is not None:
            if split == "train":
                episode_starts = episode_starts[:n_episodes + 1]
            else:
                episode_starts = episode_starts[n_episodes:]

        for start, end in zip(episode_starts, tqdm(episode_starts[1:], desc="Creating dataset")):
            episode = self.data["obs"][start:end]
            actions = self.data["actions"][start:end].reshape((-1, 1))
            self.actions = np.append(self.actions, actions[:-1], axis=0)
            self.states = np.append(self.states, episode[:-1], axis=0)
            self.next_states = np.append(self.next_states, episode[1:], axis=0)

            if source != "local":
                self.average_reward.append(self.data["rewards"][start:end].sum())

        if isinstance(self.average_reward, list):
            self.average_reward = np.mean(self.average_reward)

        assert self.states.shape[0] == self.actions.shape[0] == self.next_states.shape[0]

        self.states = torch.from_numpy(self.states)
        self.actions = torch.from_numpy(self.actions)
        self.next_states = torch.from_numpy(self.next_states)

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
        state = self.states[index]
        action = self.actions[index]
        next_state = self.next_states[index]
        return state, action, next_state
