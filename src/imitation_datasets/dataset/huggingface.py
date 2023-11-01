"""Module for HuggingFace utility functions."""
from typing import List, Dict, Any
import os
import json

import numpy as np
from numpy.lib.npyio import NpzFile
from datasets import Dataset
from tqdm import tqdm


def convert_baseline_dataset_to_dict(dataset: NpzFile) -> List[Dict[str, Any]]:
    """Convert a  NpzFile dataset into a dict dataset for the baseline function.

    Args:
        dataset (NpzFile): dataset numpy file.

    Returns:
        dataset (List[Dict[str, Any]]): converted dataset.
    """
    converted = []
    for index in tqdm(range(dataset["obs"].shape[0]), desc="Converting to dict"):
        row = {}
        for key in ["obs", "actions", "rewards", "episode_starts"]:
            row[key] = dataset[key][index].tolist()
        converted.append(row)
    return converted


def save_dataset_into_huggingface_format(dataset: List[Dict[str, Any]], path: str) -> None:
    """Save Hasmap dataset into a JSONL file.

    Args:
        dataset (List[Dict[str, Any]]): dataset in Hashmap format.
        path (str): path to save the new dataset.
    """
    with open(path, "w", encoding="utf-8") as f:
        for line in tqdm(dataset, desc="Writing into file"):
            f.write(json.dumps(line) + "\n")


def baseline_to_huggingface(dataset_path: str, new_path: str) -> None:
    """Loads baseline dataset from NpzFile, converts into a dict and save it
    into a JSONL file for upload.

    Args:
        dataset_path (str): path to the npz file.
        new_path (str): path to the new dataset.

    Raises:
        ValueError: if one of the paths does not exist.
    """
    path = "/".join(dataset_path.split("/")[:-1])
    if not os.path.exists(path):
        raise ValueError(f"'{path}' does not exist.")

    path = "/".join(new_path.split("/")[:-1])
    if not os.path.exists(path):
        raise ValueError(f"'{path}' does not exist.")

    dataset = np.load(dataset_path, allow_pickle=True)
    dataset = convert_baseline_dataset_to_dict(dataset)
    save_dataset_into_huggingface_format(dataset, new_path)


def huggingface_to_baseline(dataset: Dataset) -> Dict[str, np.ndarray]:
    """Transform a HuggingFace dataset to a format similar to the NpzFile.

    Args:
        dataset (datasets.Dataset): HuggingFace dataset.

    Returns:
        dataset (Dict[str, np.ndarray]): dataset with Baseline pattern.
    """
    obs = np.ndarray(shape=(0, len(dataset[0]["obs"])))

    if isinstance(dataset[0]["actions"], list):
        actions = np.ndarray(shape=(0, len(dataset[0]["actions"])))
    else:
        actions = []

    rewards = []
    episode_starts = []

    for row in dataset:
        _obs = np.array(row["obs"])[None]
        obs = np.append(obs, _obs, axis=0)

        if isinstance(row["actions"], list):
            _actions = np.array(row["actions"])[None]
            actions = np.append(actions, _actions, axis=0)
        else:
            actions.append(row["actions"])

        rewards.append(row["rewards"])
        episode_starts.append(row["episode_starts"])

    return {
        "obs": np.array(obs),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "episode_starts": np.array(episode_starts)
    }
