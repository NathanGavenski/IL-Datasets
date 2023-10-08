"""Module for testing registers."""
from unittest import TestCase

from abc import ABCMeta

from imitation_datasets import register
from imitation_datasets.register import *


class TestRegisters(TestCase):
    """Test cases for registers"""

    def setUp(self):
        self.registers = [reg for reg in dir(register) if "_" not in reg]

    def test_registers(self) -> None:
        """Test if registers follow pattern"""
        for environment_type in self.registers:
            for values in globals()[environment_type].values():
                assert 'name' in values.keys()
                assert 'repo_id' in values.keys()
                assert 'algo' in values.keys()
                assert 'filename' in values.keys()
                assert 'threshold' in values.keys()
                assert 'algo' in values.keys()

    def test_registers_values(self) -> None:
        """Test if register values follow types"""
        for environment_type in self.registers:
            for values in globals()[environment_type].values():
                assert isinstance(values['name'], str)
                assert isinstance(values['repo_id'], str)
                assert isinstance(values['filename'], str)
                assert isinstance(values['threshold'], (float, int))
                assert isinstance(values['algo'], ABCMeta)

    def test_environments(self) -> None:
        """Test all environment types for registers"""
        assert 'atari' in self.registers
        assert 'mujoco' in self.registers
        assert 'classic' in self.registers
