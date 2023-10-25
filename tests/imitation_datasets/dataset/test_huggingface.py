from unittest import TestCase
import os
import shutil

from datasets import load_dataset
import numpy as np

from src.imitation_datasets.experts import Experts
from src.imitation_datasets.functions import baseline_enjoy, baseline_collate
from src.imitation_datasets.utils import Context, Experiment
from src.imitation_datasets.dataset import (
    convert_baseline_dataset_to_dict,
    save_dataset_into_huggingface_format,
    baseline_to_huggingface,
    huggingface_to_baseline
)


class TestHuggingFace(TestCase):

    def setUp(self) -> None:
        self.huggingface_dataset = load_dataset("NathanGavenski/CartPole-v1", split="train")

        if not os.path.exists("./tmp/"):
            os.makedirs("./tmp/data/")
            os.makedirs("./tmp/logs/")

        baseline_enjoy(
            Experts().get_expert("cartpole"),
            "./tmp/data/",
            Context(Experiment(1, "./tmp/logs/log.txt"), 0)
        )
        files = list(os.listdir("./tmp/data/"))
        baseline_collate("./tmp/data/", files)
        self.numpy_dataset = np.load("./tmp/data/teacher.npz", allow_pickle=True)

    def tearDown(self) -> None:
        del self.huggingface_dataset
        del self.numpy_dataset

        if os.path.exists("./tmp/"):
            shutil.rmtree("./tmp/")

    def test_convert_baseline_to_dict(self) -> None:
        converted = convert_baseline_dataset_to_dict(self.numpy_dataset)
        assert self.numpy_dataset["obs"].shape[0] == len(converted)
        assert list(converted[0].keys()) == ["obs", "actions", "rewards", "episode_starts"]

    def test_save_dataset_into_huggingface_format(self) -> None:
        converted = convert_baseline_dataset_to_dict(self.numpy_dataset)
        save_dataset_into_huggingface_format(converted, "./tmp/teacher.jsonl")
        assert os.path.exists("./tmp/teacher.jsonl")

    def test_baseline_to_huggingface(self) -> None:
        converted = huggingface_to_baseline(self.huggingface_dataset)
        assert converted["obs"].shape[0] == len(self.huggingface_dataset)
        assert list(converted.keys()) == ["obs", "actions", "rewards", "episode_starts"]
