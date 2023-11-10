"""Module for utility functions

    load_hyperparameters: load hyperparameters from YAML file
    convert_hyperparameters: convert string values into other typings
    import_hyperparameters: loads and converts string values.
"""
from typing import Dict, Any
import os

import yaml


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

    with open(path, 'r', encoding='urf-8') as _file:
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
    hyperparameters['lr'] = float(hyperparameters['lr'])
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
