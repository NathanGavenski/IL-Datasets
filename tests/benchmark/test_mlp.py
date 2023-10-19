from unittest import TestCase

from src.benchmark.methods.policies.attention import Self_Attn1D
from src.benchmark.methods.policies.mlp import MLP, MlpWithAttention, MlpAttention


class TestMLP(TestCase):

    def test_init(self) -> None:
        model = MLP(2, 2)
        assert model.input.in_features == 2
        assert model.input.out_features == 8
        assert model.fc.in_features == 8
        assert model.fc.out_features == 8
        assert model.fc2.in_features == 8
        assert model.fc2.out_features == 8
        assert model.output.in_features == 8
        assert model.output.out_features == 2

        model = MLP(8, 2)
        assert model.input.in_features == 8
        assert model.input.out_features == 8 * 2
        assert model.fc.in_features == 8 * 2
        assert model.fc.out_features == 8 * 2
        assert model.fc2.in_features == 8 * 2
        assert model.fc2.out_features == 8 * 2
        assert model.output.in_features == 8 * 2
        assert model.output.out_features == 2


class TestMlpWithAttention(TestCase):

    def test_init(self) -> None:
        model = MlpWithAttention(2, 2)
        assert model.input.in_features == 2
        assert model.input.out_features == 8
        assert model.fc.in_features == 8
        assert model.fc.out_features == 8
        assert model.fc2.in_features == 8
        assert model.fc2.out_features == 8
        assert model.output.in_features == 8
        assert model.output.out_features == 2

        model = MlpWithAttention(8, 2)
        assert model.input.in_features == 8
        assert model.input.out_features == 8 * 2
        assert model.fc.in_features == 8 * 2
        assert model.fc.out_features == 8 * 2
        assert model.fc2.in_features == 8 * 2
        assert model.fc2.out_features == 8 * 2
        assert model.output.in_features == 8 * 2
        assert model.output.out_features == 2

        assert isinstance(model.attention, Self_Attn1D)
        assert isinstance(model.attention2, Self_Attn1D)


class TestMlpAttention(TestCase):

    def test_init(self) -> None:
        model = MlpAttention(2, 2)
        assert model.input.in_features == 2
        assert model.input.out_features == 8
        assert model.output.in_features == 8
        assert model.output.out_features == 2

        model = MlpAttention(8, 2)
        assert model.input.in_features == 8
        assert model.input.out_features == 8 * 2
        assert model.output.in_features == 8 * 2
        assert model.output.out_features == 2

        assert isinstance(model.attention, Self_Attn1D)
        assert isinstance(model.attention2, Self_Attn1D)
