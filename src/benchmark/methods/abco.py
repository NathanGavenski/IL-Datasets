"""Module for Augmented Behavioural Cloning from Observation"""
from collections import defaultdict
from numbers import Number
from typing import List, Union, Dict, Tuple

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from gymnasium import Env
import numpy as np
import torch
from torch.utils.data import DataLoader
from imitation_datasets.utils import GymWrapper
from imitation_datasets.dataset.metrics import average_episodic_reward, performance

from .bco import BCO
from .method import Metrics
from .utils import reached_goal


path = "/".join(__file__.split("/")[:-1])
CONFIG_FILE = f"{path}/config/abco.yaml"


class ABCO(BCO):
    """Augmented Behavioural Cloning from Observation method based on (Monteiro et. al., 2020)"""

    __version__ = "1.0.0"
    __author__ = "Monteiro et. al."
    __method_name__ = "Augmented Behavioural Cloning from Observation"

    def __init__(
        self,
        environment: Env,
        enjoy_criteria: int = 100,
        verbose: bool = False,
        config_file: str = None
    ) -> None:
        if config_file is None:
            config_file = CONFIG_FILE
        super().__init__(environment, enjoy_criteria, verbose, config_file)
        self.save_path = f"./tmp/abco/{self.environment_name}/"

    def train(
        self,
        n_epochs: int,
        train_dataset: Dict[str, DataLoader],
        eval_dataset: Dict[str, DataLoader] = None,
        folder: str = None
    ) -> Self:
        if folder is None:
            folder = f"../benchmark_results/abco/{self.environment_name}"

        super().train(
            n_epochs,
            train_dataset,
            eval_dataset,
            folder
        )
        return self

    def _append_samples(self, train_dataset: DataLoader) -> DataLoader:
        """Append samples to DataLoader.

        Args:
            train_dataset (DataLoader): current train dataset.

        Returns:
            train_dataset (DataLoader): new train dataset.
        """
        metrics, i_pos = self._enjoy(return_ipos=True)
        i_pos_ratio = metrics.get('success_rate', 0)
        idm_ratio = 1 - i_pos_ratio

        if i_pos_ratio == 0:
            return train_dataset

        i_pos_size = i_pos["states"].shape[0]
        idm_size = train_dataset['idm_dataset'].dataset.states.shape[0]
        i_pos_k = max(0, int(i_pos_size * i_pos_ratio))
        idm_k = max(0, int(idm_size * idm_ratio))

        i_pos_idx = torch.multinomial(torch.tensor(range(i_pos_size)).float(), i_pos_k)
        try:
            idm_idx = torch.multinomial(torch.tensor(range(idm_size)).float(), idm_k)
        except RuntimeError:
            idm_idx = []

        train_dataset['idm_dataset'].dataset.states = torch.cat((
            train_dataset['idm_dataset'].dataset.states[idm_idx],
            torch.from_numpy(i_pos['states'])[i_pos_idx]),
            dim=0
        )
        train_dataset['idm_dataset'].dataset.next_states = torch.cat((
            train_dataset['idm_dataset'].dataset.next_states[idm_idx],
            torch.from_numpy(i_pos['next_states'])[i_pos_idx]),
            dim=0
        )
        train_dataset['idm_dataset'].dataset.actions = torch.cat((
            train_dataset['idm_dataset'].dataset.actions[idm_idx],
            torch.from_numpy(i_pos['actions'].reshape((-1, 1)))[i_pos_idx]),
            dim=0
        )
        return train_dataset

    def _enjoy(
        self,
        render: bool = False,
        teacher_reward: Number = None,
        random_reward: Number = None,
        return_ipos: bool = False,
    ) -> Union[Metrics, Tuple[Metrics, Dict[str, List[float]]]]:
        """Function for evaluation of the policy in the environment

        Args:
            render (bool): Whether it should render. Defaults to False.
            teacher_reward (Number): reward for teacher policy.
            random_reward (Number): reward for a random policy.
            return_ipos (bool): whether it should return data to append to I_pos.

        Returns:
            Metrics:
                aer (Number): average reward for 100 episodes.
                aer_std (Number): standard deviation for aer.
                performance (Number): if teacher_reward and random_reward are
                    informed than the performance metric is calculated.
                perforamance_std (Number): standard deviation for performance.
                success_rate (float): percentage that the agent reached the goal.
            I_pos:
                states (List[Number]): states before action.
                actions (List[Number]): action given states.
                next_states (List[Number]): next state given states and actions.
        """
        environment = GymWrapper(self.environment)
        average_reward = []
        i_pos = defaultdict(list)
        success_rate = []

        for _ in range(100):
            done = False
            obs = environment.reset()
            accumulated_reward = 0
            goal = False
            while not done:
                if render:
                    environment.render()
                action = self.predict(obs)

                i_pos['states'].append(obs)
                i_pos['actions'].append(action)

                gym_return = environment.step(action)
                obs, reward, done, *_ = gym_return
                accumulated_reward += reward
                goal |= reached_goal(self.environment_name,
                                     gym_return, accumulated_reward)

                i_pos['next_states'].append(obs)
            average_reward.append(accumulated_reward)
            success_rate.append(goal)

        metrics = average_episodic_reward(average_reward)
        if teacher_reward is not None and random_reward is not None:
            metrics.update(performance(
                average_reward, teacher_reward, random_reward))
        metrics['success_rate'] = np.mean(success_rate)

        i_pos = {key: np.array(value) for key, value in i_pos.items()}

        if return_ipos:
            return metrics, i_pos
        return metrics
