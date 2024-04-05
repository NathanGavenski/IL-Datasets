from unittest import TestCase
import os

import pytest
import torch
from torchvision import models, transforms

from src.benchmark.methods.policies.attention import SelfAttn2D
from src.benchmark.methods.policies.cnn import (
    Empty,
    convert_to_bw,
    convert_to_n_channels,
    normalize_imagenet,
    CNN,
    Resnet,
    ResnetWithAttention
)


github = os.getenv("SERVER")
github = bool(int(github)) if github is not None else False


class TestEmpty(TestCase):

    def test_empty_layer(self) -> None:
        empty = Empty()
        assert len(list(empty.parameters())) == 0

    def test_empty_forward(self) -> None:
        empty = Empty()
        x = torch.ones((3, 32, 32))
        assert (empty(x) == x).all()


class TestConvertLayers(TestCase):

    def setUp(self) -> None:
        self.model = models.resnet18(pretrained=False)

    def test_convert_bw(self) -> None:
        model = convert_to_bw(self.model)
        assert model.conv1.in_channels == 1

    def test_convert_n_channels(self) -> None:
        model = convert_to_n_channels(self.model, 4)
        assert model.conv1.in_channels == 4
        assert not model.conv1.bias


class TestNormalize(TestCase):

    @pytest.mark.skipif(github, reason="inconsistent results")
    def test_normalize_rgb(self) -> None:
        tensor = torch.Tensor(size=(1, 3, 64, 64))
        normalized = normalize_imagenet(tensor, "rgb")

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        manual_normalized = transforms.Normalize(mean, std)(tensor)

        assert (normalized == manual_normalized).all()

    @pytest.mark.skipif(github, reason="inconsistent results")
    def test_normalize_bw(self) -> None:
        tensor = torch.Tensor(size=(1, 1, 64, 64))
        normalized = normalize_imagenet(tensor, "bw")

        mean = 0.44531356896770125
        std = 0.2692461874154524
        manual_normalized = transforms.Normalize(mean, std)(tensor)

        assert (normalized == manual_normalized).all()


class TestCNN(TestCase):

    def setUp(self) -> None:
        self.model = CNN()

    def test_cnn_layers(self) -> None:
        assert self.model.block0[0].in_channels == 4
        assert self.model.block0[0].out_channels == 32
        assert self.model.block0[0].kernel_size == (8, 8)
        assert self.model.block0[0].stride == (4, 4)

        assert self.model.block1[0].in_channels == 32
        assert self.model.block1[0].out_channels == 64
        assert self.model.block1[0].kernel_size == (4, 4)
        assert self.model.block1[0].stride == (2, 2)

        assert self.model.block2[0].in_channels == 64
        assert self.model.block2[0].out_channels == 64
        assert self.model.block2[0].kernel_size == (3, 3)
        assert self.model.block2[0].stride == (1, 1)

    def test_cnn_forward(self) -> None:
        tensor = torch.zeros((1, 4, 84, 84))
        output = self.model(tensor)
        assert list(output.shape) == [1, 3136]


class TestResnet(TestCase):

    def test_resnet_modes(self) -> None:
        bw_model = Resnet((84, 84, 1))
        assert bw_model.model.conv1.in_channels == 1

        rgb_model = Resnet((84, 84, 3))
        assert rgb_model.model.conv1.in_channels == 3

        atari_model = Resnet((84, 84, 4))
        assert atari_model.model.conv1.in_channels == 4

    def test_resnetattention_modes(self) -> None:
        bw_model = ResnetWithAttention((84, 84, 1))
        assert bw_model.model.conv1.in_channels == 1
        assert isinstance(bw_model.model.layer1[-1], SelfAttn2D)

        rgb_model = ResnetWithAttention((84, 84, 3))
        assert rgb_model.model.conv1.in_channels == 3
        assert isinstance(rgb_model.model.layer1[-1], SelfAttn2D)

        atari_model = ResnetWithAttention((84, 84, 4))
        assert atari_model.model.conv1.in_channels == 4
        assert isinstance(atari_model.model.layer1[-1], SelfAttn2D)
