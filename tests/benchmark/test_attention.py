from unittest import TestCase

import torch
from torch import nn

from src.benchmark.methods.policies.attention import SelfAttn1D, SelfAttn2D


class TestSelfAtten2D(TestCase):

    def test_init(self) -> None:
        attention = SelfAttn2D(3)
        assert attention.chanel_in == 3
        assert attention.gamma == nn.Parameter(torch.zeros(1))
        assert isinstance(attention.softmax, nn.Softmax)

    def test_forward(self) -> None:
        attention_layer = SelfAttn2D(64)
        feature_map = torch.zeros((1, 64, 32, 32))
        result = attention_layer(feature_map, True)
        assert len(result) == 2

        out, attention = result
        assert isinstance(out, torch.Tensor)
        assert isinstance(attention, torch.Tensor)
        assert feature_map.shape == out.shape
        attention_size = feature_map.size(2) * feature_map.size(3)
        assert list(attention.shape) == [1, attention_size, attention_size]

        result = attention_layer(feature_map)
        assert len(result) == 1

        out = result
        assert isinstance(out, torch.Tensor)
        assert isinstance(attention, torch.Tensor)
        assert feature_map.shape == out.shape


class TestSelfAtten1D(TestCase):

    def test_init(self) -> None:
        attention = SelfAttn1D(3)
        assert attention.chanel_in == 3
        assert attention.gamma == nn.Parameter(torch.zeros(1))
        assert isinstance(attention.softmax, nn.Softmax)

    def test_forward(self) -> None:
        attention_layer = SelfAttn1D(1080)
        feature_map = torch.zeros((1, 1080))
        result = attention_layer(feature_map, True)
        assert len(result) == 2

        out, attention = result
        assert isinstance(out, torch.Tensor)
        assert isinstance(attention, torch.Tensor)
        assert feature_map.shape == out.shape

        result = attention_layer(feature_map)
        assert len(result) == 1

        out = result
        assert isinstance(out, torch.Tensor)
        assert isinstance(attention, torch.Tensor)
        assert feature_map.shape == out.shape
