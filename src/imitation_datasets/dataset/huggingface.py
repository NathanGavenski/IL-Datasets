"""Module for HuggingFace utility functions."""
from typing import List, Dict, Any
import os
import json

import numpy as np
import pandas as pd
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
    dataframe = pd.DataFrame.from_dict(
        {key: dataset[key].tolist() for key in ["obs", "actions", "rewards", "episode_starts"]},
        orient="index"
    ).T
    return dataframe.to_dict(orient="records")


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
    dataframe = dataset.to_pandas()

    return {
        "obs": np.array(dataframe["obs"].to_list()),
        "actions": np.array(dataframe["actions"].to_list()),
        "rewards": np.array(dataframe["rewards"].to_list()),
        "episode_starts": np.array(dataframe["episode_starts"].to_list())
    }
