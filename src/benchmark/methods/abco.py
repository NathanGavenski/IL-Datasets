"""Module for Augmented Behavioural Cloning from Observation"""
import torch
from torch.utils.data import DataLoader
from gymnasium import Env

from .policies.mlp import MlpWithAttention
from .bco import BCO


CONFIG_FILE = "./src/benchmark/methods/config/abco.yaml"


class ABCO(BCO):
    """Augmented Behavioural Cloning from Observation method based on (Monteiro et. al., 2020)"""

    __version__ = "1.0.0"
    __author__ = "Monteiro et. al."
    __method_name__ = "Augmented Behavioural Cloning from Observation"

    def __init__(self, environment: Env, enjoy_criteria: int = 100, verbose: bool = False) -> None:
        super().__init__(environment, enjoy_criteria, verbose)
        self.save_path = f"./tmp/abco/{self.environment_name}/"

        self.idm = MlpWithAttention(self.observation_size * 2, self.action_size)
        self.idm_optimizer = open.Adam(self.idm.parameters(), lr=self.hyperparameters['idm_lr'])
        self.idm_loss = nn.CrossEntropyLoss() if self.discrete else nn.MSELoss()

    def _append_samples(self, train_dataset: DataLoader) -> DataLoader:
        _, i_pos = self._enjoy(return_ipos=True)
        train_dataset['idm_dataset'].dataset.states = torch.cat((
            train_dataset['idm_dataset'].dataset.states,
            torch.from_numpy(i_pos['states'])),
            dim=0
        )
        train_dataset['idm_dataset'].dataset.next_states = torch.cat((
            train_dataset['idm_dataset'].dataset.next_states,
            torch.from_numpy(i_pos['next_states'])),
            dim=0
        )
        train_dataset['idm_dataset'].dataset.actions = torch.cat((
            train_dataset['idm_dataset'].dataset.actions,
            torch.from_numpy(i_pos['actions'].reshape((-1, 1)))),
            dim=0
        )
        return train_dataset
