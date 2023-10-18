from unittest import TestCase
import os
import shutil

import numpy as np

from src.imitation_datasets.experts import Experts
from src.imitation_datasets.functions import enjoy, baseline_enjoy, collate, baseline_collate
from src.imitation_datasets.utils import Context, Experiment


class FunctionsTest(TestCase):

    def setUp(self) -> None:
        self.expert = Experts().get_expert("cartpole")
        self.tmp_folder = "./tmp/"
        self.data_folder = f"{self.tmp_folder}data/"
        if not os.path.exists(self.tmp_folder):
            os.makedirs(self.tmp_folder)
            os.makedirs(f"{self.tmp_folder}data/")
            os.makedirs(f"{self.tmp_folder}logs/")
        self.experiment = Experiment(1, f"{self.tmp_folder}logs/logs.txt")

    def tearDown(self) -> None:
        del self.expert
        if os.path.exists(self.tmp_folder):
            shutil.rmtree(self.tmp_folder)

    def test_simple_enjoy(self) -> None:
        context = Context(self.experiment, 0)
        result = enjoy(
            self.expert,
            self.data_folder,
            context
        )
        assert result  # Assert that the teacher succeeded

        path = f"{self.data_folder}/{context.index}.npz"
        assert os.path.exists(path)

        data = np.load(path)
        assert list(data.keys()) == ["states", "actions"]
        for value in data.values():
            assert isinstance(value, np.ndarray)

    def test_baseline_enjoy(self) -> None:
        context = Context(self.experiment, 0)
        result = baseline_enjoy(
            self.expert,
            self.data_folder,
            context
        )
        assert result  # Assert that the teacher succeeded

        path = f"{self.data_folder}/{context.index}.npz"
        assert os.path.exists(path)

        data = np.load(path)
        assert list(data.keys()) == ["obs", "actions", "rewards", "episode_returns"]
        for value in data.values():
            assert isinstance(value, np.ndarray)
        assert len(data["episode_returns"]) == 1

    def test_simple_collate(self) -> None:
        context = Context(self.experiment, 0)
        result = enjoy(
            self.expert,
            self.data_folder,
            context
        )
        assert result  # Assert that the teacher succeeded

        files = list(os.listdir(self.data_folder))
        collate(self.data_folder, files)
        assert os.path.exists(f"{self.data_folder}teacher.npz")

        teacher = np.load(f"{self.data_folder}teacher.npz")
        assert list(teacher.keys()) == ["states","actions","episode_starts"]

        for values in teacher.values():
            assert isinstance(values, np.ndarray)

        assert teacher["episode_starts"][0]
        assert not all(teacher["episode_starts"][1:])

    def test_baseline_collate(self) -> None:
        context = Context(self.experiment, 0)
        result = baseline_enjoy(
            self.expert,
            self.data_folder,
            context
        )
        assert result  # Assert that the teacher succeeded

        files = list(os.listdir(self.data_folder))
        baseline_collate(self.data_folder, files)
        assert os.path.exists(f"{self.data_folder}teacher.npz")

        keys = ["obs","actions", "rewards", "episode_returns", "episode_starts"]
        teacher = np.load(f"{self.data_folder}teacher.npz")
        assert list(teacher.keys()) == keys

        for values in teacher.values():
            assert isinstance(values, np.ndarray)

        assert teacher["episode_starts"][0]
        assert not all(teacher["episode_starts"][1:])
