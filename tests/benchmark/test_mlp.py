from unittest import TestCase

from src.benchmark.methods.policies.attention import SelfAttn1D
from src.benchmark.methods.policies.mlp import MLP, MlpWithAttention, MlpAttention


class TestMLP(TestCase):

    def test_init(self) -> None:
        model = MLP(2, 2)
        assert model.layers[0].in_features == 2
        assert model.layers[0].out_features == 32
        assert model.layers[2].in_features == 32
        assert model.layers[2].out_features == 32
        assert model.layers[4].in_features == 32
        assert model.layers[4].out_features == 32
        assert model.layers[6].in_features == 32
        assert model.layers[6].out_features == 2

        model = MLP(30, 2)
        assert model.layers[0].in_features == 30
        assert model.layers[0].out_features == 30 * 2
        assert model.layers[2].in_features == 30 * 2
        assert model.layers[2].out_features == 30 * 2
        assert model.layers[4].in_features == 30 * 2
        assert model.layers[4].out_features == 30 * 2
        assert model.layers[6].in_features == 30 * 2
        assert model.layers[6].out_features == 2


class TestMlpWithAttention(TestCase):

    def test_init(self) -> None:
        model = MlpWithAttention(2, 2)
        assert model.layers[0].in_features == 2
        assert model.layers[0].out_features == 8
        assert model.layers[3].in_features == 8
        assert model.layers[3].out_features == 8
        assert model.layers[6].in_features == 8
        assert model.layers[6].out_features == 8
        assert model.layers[8].in_features == 8
        assert model.layers[8].out_features == 8
        assert model.layers[10].in_features == 8
        assert model.layers[10].out_features == 2

        model = MlpWithAttention(8, 2)
        assert model.layers[0].in_features == 8
        assert model.layers[0].out_features == 8 * 2
        assert model.layers[3].in_features == 8 * 2
        assert model.layers[3].out_features == 8 * 2
        assert model.layers[6].in_features == 8 * 2
        assert model.layers[6].out_features == 8 * 2
        assert model.layers[8].in_features == 8 * 2
        assert model.layers[8].out_features == 8 * 2
        assert model.layers[10].in_features == 8 * 2
        assert model.layers[10].out_features == 2

        assert isinstance(model.layers[2], SelfAttn1D)
        assert isinstance(model.layers[5], SelfAttn1D)


class TestMlpAttention(TestCase):

    def test_init(self) -> None:
        model = MlpAttention(2, 2)
        assert model.layers[0].in_features == 2
        assert model.layers[0].out_features == 8
        assert model.layers[6].in_features == 8
        assert model.layers[6].out_features == 2

        model = MlpAttention(8, 2)
        assert model.layers[0].in_features == 8
        assert model.layers[0].out_features == 8 * 2
        assert model.layers[6].in_features == 8 * 2
        assert model.layers[6].out_features == 2

        assert isinstance(model.layers[2], SelfAttn1D)
        assert isinstance(model.layers[4], SelfAttn1D)
