from unittest import TestCase

import numpy as np
import pytest
import torch

from src.imitation_datasets.dataset.metrics import (
    performance,
    average_episodic_reward,
    accuracy
)


class TestMetrics(TestCase):

    def test_performance(self) -> None:
        metric = performance(50, 100, 0)
        assert metric == {"performance": 0.5, "performance_std": 0}

        metric = performance([100, 50, 0], 100, 0)
        assert metric == {'performance': 0.5, 'performance_std': 0.408248290463863}

        metric = performance(np.array([100, 50, 0]), 100, 0)
        assert metric == {'performance': 0.5, 'performance_std': 0.408248290463863}

        with pytest.raises(ValueError):
            performance(50, 0, 100)

        with pytest.raises(ValueError):
            performance(50, [100, 50], 0)

        with pytest.raises(ValueError):
            performance(50, np.ndarray([100, 50]), 0)

        with pytest.raises(ValueError):
            performance(50, 100, [100, 50])

        with pytest.raises(ValueError):
            performance(50, 100, np.ndarray([100, 50]))

    def test_average_episodic_reward(self) -> None:
        metric = average_episodic_reward([100, 50, 0])
        assert metric == {'aer': 50.0, 'aer_std': 40.824829046386306}

        metric = average_episodic_reward(np.array([100, 50, 0]))
        assert metric == {'aer': 50.0, 'aer_std': 40.824829046386306}

    def test_accuracy(self) -> None:
        logits = torch.Tensor([[0.5, 0.2], [0.3, 0.7], [0.1, -0.25]])

        metric = accuracy(logits, torch.Tensor([0, 1, 0]))
        assert metric == 100

        metric = accuracy(logits, torch.Tensor([1, 0, 1]))
        assert metric == 0

        with pytest.raises(ValueError):
            accuracy(logits, [0, 1, 0])

        with pytest.raises(ValueError):
            accuracy(
                [[0.5, 0.2], [0.3, 0.7], [0.1, -0.25]],
                torch.Tensor([0, 1, 0])
            )

        with pytest.raises(ValueError):
            accuracy(
                logits.view((-1)),
                torch.Tensor([0, 1, 0])
            )

        with pytest.raises(ValueError):
            accuracy(
                logits,
                torch.Tensor([[0], [1], [0]])
            )
