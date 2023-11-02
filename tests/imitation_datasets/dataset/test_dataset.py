from unittest import TestCase
import os
import shutil

from pytest import raises

from src.imitation_datasets.experts import Experts
from src.imitation_datasets.functions import baseline_enjoy, baseline_collate
from src.imitation_datasets.utils import Context, Experiment
from src.imitation_datasets.dataset import BaselineDataset


class TestDataset(TestCase):

    def tearDown(self) -> None:
        if os.path.exists("./tmp/"):
            shutil.rmtree("./tmp/")

    def test_local_file(self) -> None:
        if not os.path.exists("./tmp/"):
            os.makedirs("./tmp/data/")
            os.makedirs("./tmp/logs/")

        for index in range(5):
            baseline_enjoy(
                Experts().get_expert("cartpole"),
                "./tmp/data/",
                Context(Experiment(5, "./tmp/logs/log.txt"), index)
            )
        files = list(os.listdir("./tmp/data/"))
        baseline_collate("./tmp/data/", files)

        dataset = BaselineDataset('./tmp/data/teacher.npz')
        assert len(dataset) == 500 * 5 - 5
        assert dataset.states.shape[0] == 500 * 5 - 5
        assert dataset.next_states.shape[0] == 500 * 5 - 5
        assert dataset.actions.shape[0] == 500 * 5 - 5
        assert dataset.average_reward == 500

        state, action, next_state = dataset[0]
        assert len(state.size()) == 1
        assert len(action.size()) == 1
        assert len(next_state.size()) == 1
        assert state.size(0) == 4
        assert action.size(0) == 1
        assert next_state.size(0) == 4

        dataset = BaselineDataset('./tmp/data/teacher.npz', n_episodes=2)
        assert len(dataset) == 500 * 2 - 2
        assert dataset.states.shape[0] == 500 * 2 - 2
        assert dataset.next_states.shape[0] == 500 * 2 - 2
        assert dataset.actions.shape[0] == 500 * 2 - 2
        assert dataset.average_reward == 500

        state, action, next_state = dataset[0]
        assert len(state.size()) == 1
        assert len(action.size()) == 1
        assert len(next_state.size()) == 1
        assert state.size(0) == 4
        assert action.size(0) == 1
        assert next_state.size(0) == 4

        with raises(ValueError):
            BaselineDataset('./tmp/teacher.npz')

    def test_huggingface_file(self) -> None:
        dataset = BaselineDataset('NathanGavenski/CartPole-v1', source='huggingface', n_episodes=10)
        assert len(dataset) == 500 * 10 - 10
        assert dataset.states.shape[0] == 500 * 10 - 10
        assert dataset.next_states.shape[0] == 500 * 10 - 10
        assert dataset.actions.shape[0] == 500 * 10 - 10
        assert dataset.average_reward == 500

        state, action, next_state = dataset[0]
        assert len(state.size()) == 1
        assert len(action.size()) == 1
        assert len(next_state.size()) == 1
        assert state.size(0) == 4
        assert action.size(0) == 1
        assert next_state.size(0) == 4

        dataset = BaselineDataset(
            'NathanGavenski/CartPole-v1',
            source='huggingface',
            n_episodes=2
        )
        assert len(dataset) == 500 * 2 - 2
        assert dataset.states.shape[0] == 500 * 2 - 2
        assert dataset.next_states.shape[0] == 500 * 2 - 2
        assert dataset.actions.shape[0] == 500 * 2 - 2
        assert dataset.average_reward == 500

        state, action, next_state = dataset[0]
        assert len(state.size()) == 1
        assert len(action.size()) == 1
        assert len(next_state.size()) == 1
        assert state.size(0) == 4
        assert action.size(0) == 1
        assert next_state.size(0) == 4
