"""Module for metrics"""
from numbers import Number
from typing import Union, List, Dict

import numpy as np
from torch import Tensor, argmax


def performance(
    agent_reward: Union[Number, List, np.ndarray],
    teacher_reward: Number,
    random_reward: Number
) -> Dict[str, Number]:
    """Compute the performance for the agent. Performance normalises between
    random and expert policies rewards, where performance 0 corresponds to
    random policy performance, and 1 are for expert policy performance.

    performance = (X - X_min) / (X_max - X_min),

    where X_min is the random_reward, and X_max is the teacher_reward.

    Args:
        agent_reward (Number): agent accumulated reward.
        teacher_reward (Number): teacher accumulated reward.
        random_reward (Number): random agent accumulated reward.

    Raises:
        ValueError: if the teacher reward is inferior to the random agent.
        ValueError: Teacher and Random rewards should be Numbers.

    Returns:
        performance (Number): performance metric.
    """
    if isinstance(teacher_reward, (list, np.ndarray)):
        raise ValueError("Teacher reward should not be a list")

    if isinstance(random_reward, (list, np.ndarray)):
        raise ValueError("Random reward should not be a list")

    if teacher_reward < random_reward:
        raise ValueError("Random reward should lesser than the teacher's.")

    if isinstance(agent_reward, list):
        agent_reward = np.array(agent_reward)

    perf = (agent_reward - random_reward) / (teacher_reward - random_reward)
    if isinstance(perf, np.ndarray):
        return {"performance": perf.mean(), "performance_std": perf.std()}
    return {"performance": perf, "performance_std": 0}


def average_episodic_reward(agent_reward: List[Number]) -> Dict[str, Number]:
    """Compute the average episodic reward for the agent. AER is the average
    of 'n' episodes for each agent in each environment.

    Args:
        agent_reward (List[Number]): list of each episode accumulated reward.

    Returns:
        AER (Number): average episodic reward metric.
    """
    if isinstance(agent_reward, list):
        agent_reward = np.array(agent_reward)
    return {"aer": agent_reward.mean(), "aer_std": agent_reward.std()}


def accuracy(prediction: Tensor, ground_truth: Tensor) -> Number:
    """Compute the accuracy for a model. The accuracy returned is the percentage from 0 to 100.

    Args:
        prediction (torch.Tensor): logits from a model.
        ground_truth (torch.Tensor): ground truth class.

    Raises:
        ValueError: if predictions and ground_truth are not torch.Tensor.
        ValueError: if predictions are not two dimensional.
        ValueError: if ground_truth is not one dimensional.

    Returns:
        accuracy (Number): accuracy between 0 and 100 for a model.
    """

    if not isinstance(prediction, Tensor) or not isinstance(ground_truth, Tensor):
        raise ValueError("'prediction' and 'ground truth' should be a tensor")

    if len(prediction.size()) != 2:
        raise ValueError("'prediction' and 'ground truth' need to be 2 dimensional.")

    if len(ground_truth.size()) != 1:
        raise ValueError("'ground truth' need to be 1 dimensional.")

    return ((argmax(prediction, 1) == ground_truth).sum().item() / ground_truth.size(0)) * 100
