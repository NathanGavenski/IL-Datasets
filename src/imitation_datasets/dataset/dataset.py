"""Module for datasets"""
import io
import os
from typing import Tuple, Callable, Dict
import warnings

from datasets import load_dataset
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from tqdm import tqdm
from PIL import Image

from .huggingface import huggingface_to_baseline


def fn_create_dataset(
    data: Dict[str, np.ndarray],
    n_episodes: int = None,
    split: str = "train",
    get_reward: bool = False,
) -> Tuple[torch.Tensor]:
    try:
        action_shape = 1
        if len(data.get("actions").shape) > 1:
            action_shape = data.get("actions").shape[-1]
    except KeyError:
        raise AttributeError("Dataset should contain 'actions' key")

    states = []
    next_states = []
    actions = []
    if get_reward:
        rewards = []

    try:
        episode_starts = list(np.where(data.get("episode_starts") == 1)[0])
        episode_starts.append(len(data.get("episode_starts")))
    except KeyError:
        raise AttributeError("Dataset should contain 'episode_starts' key")

    if n_episodes is not None:
        episode_starts = episode_starts[:n_episodes + 1]
        if split != "train":
            episode_starts = episode_starts[n_episodes:]

    episode_end = tqdm(episode_starts[1:], desc="Creating dataset")
    for start, end in zip(episode_starts, episode_end):
        episode = data.get("obs")[start:end]
        ep_actions = data.get("actions")[start:end][:-1]
        ep_actions = ep_actions.reshape((-1, action_shape))

        states.append(episode[:-1])
        next_states.append(episode[1:])
        actions.append(ep_actions)
        if get_reward:
            ep_rewards = data.get("rewards")[start:end][:-1]
            rewards.append(ep_rewards)

    states = np.concatenate(states, axis=0)
    next_states = np.concatenate(next_states, axis=0)
    if not isinstance(states[0][0], (str, dict)):
        states = torch.from_numpy(states)
        next_states = torch.from_numpy(next_states)
    actions = torch.from_numpy(np.concatenate(actions, axis=0))

    if get_reward:
        rewards = torch.from_numpy(np.concatenate(rewards, axis=0))
        return states, actions, next_states, rewards
    return states, actions, next_states


def fn_compute_reward(data: Dict[str, np.ndarray]) -> float:
    if "rewards" not in data.keys():
        warnings.warn("No 'rewards' key found, no average reward computed")
        return None

    episodes = len(list(np.where(data.get("episode_starts") == 1)[0]))
    return data.get("rewards").sum() / episodes


class BaselineDataset(Dataset):
    """Teacher dataset for IL methods."""

    def __init__(
        self,
        path: str,
        source: str = "local",
        hf_split: str = "train",
        split: str = "train",
        n_episodes: int = None,
        transform: Callable[[torch.Tensor], torch.Tensor] = None,
        create_dataset: bool = True
    ) -> None:
        """Initialize dataset.

        Args:
            path (Str): path to the dataset.
            source (str): whether is a HuggingFace or a local dataset.
                Defaults to 'local'.
            hf_split (str): HuggingFace split to use. Defaults to 'train'.
            split (str): split to use. Defaults to 'train'.
            n_episodes (int): number of episodes to use. Defaults to None.
            transform (Callable[[torch.Tensor], torch.Tensor]): transform to apply to the data.
                Defaults to None.

        Raises:
            ValueError: if path does not exist.
        """
        if transform is not None:
            self.transform = transform
        else:
            self.transform = ToTensor()

        if source == "local" and not os.path.exists(path):
            raise ValueError(f"No dataset at: {path}")

        if source == "local":
            self.data = np.load(path, allow_pickle=True)
            self.average_reward = np.mean(self.data["episode_returns"])
        else:
            dataset = load_dataset(path, split=hf_split)
            self.data = huggingface_to_baseline(dataset)
            if len(self.data["obs"].shape) == 1:
                self.data["obs"] = self.data["obs"].reshape((-1, 1))
            self.average_reward = []

        if create_dataset:
            self.states, self.actions, self.next_states = fn_create_dataset(
                self.data, n_episodes, split
            )
            if source != "local":
                self.average_reward = fn_compute_reward(self.data)

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
        next_state = self.next_states[index]
        if isinstance(state[0], str):
            state = self.transform(Image.open(state[0]))
            next_state = self.transform(Image.open(next_state[0]))
        elif isinstance(state[0], dict):
            state = self.transform(Image.open(io.BytesIO(state[0]["bytes"])))
            next_state = self.transform(Image.open(io.BytesIO(next_state[0]["bytes"])))

        action = self.actions[index]
        return state, action, next_state


class IRLDataset(BaselineDataset):
    def __init__(
        self,
        path: str,
        source: str = "local",
        hf_split: str = "train",
        split: str = "train",
        n_episodes: int = None,
        transform: Callable[[torch.Tensor], torch.Tensor] = None
    ) -> None:
        super().__init__(path, source, hf_split, split, n_episodes, transform, False)
        if "rewards" not in self.data.keys():
            raise AttributeError("IRLDataset requires 'rewards' key to be present")

        self.states, self.actions, self.next_states, self.rewards = fn_create_dataset(
            self.data, n_episodes, split, get_reward=True
        )
        if source != "local":
            self.average_reward = fn_compute_reward(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        """Get item from dataset.

        Args:
            index (int): index.

        Returns:
            state (torch.Tensor): state for timestep t.
            action (torch.Tensor): action for timestep t.
            next_state (torch.Tensor): state for timestep t + 1.
            reward (torch.Tensor): reward for timestep t.
        """
        state, action, next_state = super().__getitem__(index)
        return state, action, next_state, self.rewards[index]
