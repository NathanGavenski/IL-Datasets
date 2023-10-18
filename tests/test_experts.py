import os
import shutil
from unittest import TestCase

import pytest

from src.imitation_datasets.experts import Experts
from src.imitation_datasets.functions import enjoy
from src.imitation_datasets.utils import Context, Experiment


github = os.getenv("SERVER")
github = bool(int(github)) if github is not None else False


class TestExperts(TestCase):
    def setUp(self) -> None:
        self.expert = Experts().get_expert("cartpole")
        self.tmp_folder = "./tmp/"
        self.data_folder = f"{self.tmp_folder}data/"
        if not os.path.exists(self.tmp_folder):
            os.makedirs(self.tmp_folder)
            os.makedirs(f"{self.tmp_folder}data/")
            os.makedirs(f"{self.tmp_folder}logs/")
        self.experiment = Experiment(1, f"{self.tmp_folder}logs/logs.txt")
        self.context = Context(self.experiment, 0)

    def tearDown(self) -> None:
        del self.expert
        if os.path.exists(self.tmp_folder):
            shutil.rmtree(self.tmp_folder)

    def test_experts_load(self) -> None:
        for environment in Experts.experts.keys():
            print(environment)
            Experts.get_expert(environment).load()

    @pytest.mark.skipif(github, reason="to not take a lot of time server wise")
    def test_experts_performance(self) -> None:
        for environment in Experts.experts.keys():
            result = enjoy(
                Experts.get_expert(environment),
                self.data_folder,
                self.context
            )
            assert result
