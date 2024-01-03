"""Module for Augmented Behavioural Cloning from Observation"""
from numbers import Number
from typing import List, Union, Dict

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from gymnasium import Env
import numpy as np
import torch
from torch.utils.data import DataLoader

from imitation_datasets.dataset.metrics import accuracy as accuracy_fn
from .abco import ABCO
from .method import Metrics


CONFIG_FILE = "./src/benchmark/methods/config/iupe.yaml"


class IUPE(ABCO):
    """Imitating Unknown Policies via Exploration method based on (Gavenski et. al., 2020)"""

    __version__ = "1.0.0"
    __author__ = "Gavenski et. al."
    __method_name__ = "Imitating Unknown Policies via Exploration"

    def __init__(self, environment: Env, enjoy_criteria: int = 100, verbose: bool = False) -> None:
        super().__init__(environment, enjoy_criteria, verbose)
        self.save_path = f"./tmp/iupe/{self.environment_name}/"
        self.is_training = False

    def predict(self, obs: Union[np.ndarray, torch.Tensor]) -> Union[List[Number], Number]:
        """Predict method.

        Args:
            obs (Union[np.ndarray, torch.Tensor]): input observation.

        Returns:
            action (Union[List[Number], Number): predicted action.
        """
        self.policy.eval()

        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)

            if len(obs.shape) == 1:
                obs = obs[None]

        obs = obs.to(self.device)

        with torch.no_grad():
            actions = self.forward(obs)
            actions = actions[0]

            if self.discrete:
                if self.is_training:
                    classes = np.arange(self.action_size)
                    prob = torch.nn.functional.softmax(actions, dim=0).cpu().detach().numpy()
                    actions = np.random.choice(classes, p=prob)
                    return actions
                return torch.argmax(actions).cpu().numpy()
            return actions.cpu().numpy()

    def train(
        self,
        n_epochs: int,
        train_dataset: Dict[str, DataLoader],
        eval_dataset: Dict[str, DataLoader] = None,
        folder: str = None
    ) -> Self:
        if folder is None:
            folder = f"../benchmark_results/iupe/{self.environment_name}"

        self.is_training = True

        try:
            super().train(
                n_epochs,
                train_dataset,
                eval_dataset,
                folder
            )
        finally:
            self.is_training = False

    def _train(self, idm_dataset: DataLoader, expert_dataset: DataLoader) -> Metrics:
        """Train loop.

        Args:
            dataset (DataLoader): train data.
        """
        if not self.idm.training:
            self.idm.train()

        if not self.policy.training:
            self.policy.train()

        idm_accumulated_loss = []
        idm_accumulated_accuracy = []

        accumulated_loss = []
        accumulated_accuracy = []

        for batch in idm_dataset:
            state, action, next_state = batch
            state = state.to(self.device)
            action = action.to(self.device)
            next_state = next_state.to(self.device)

            self.idm_optimizer.zero_grad()
            predictions = self.idm(torch.cat((state, next_state), dim=1))

            loss = self.idm_loss(predictions, action.squeeze().long())
            loss.backward()
            idm_accumulated_loss.append(loss.item())
            self.idm_optimizer.step()

            accuracy: Number = None
            if self.discrete:
                accuracy = accuracy_fn(predictions, action.squeeze())
            else:
                accuracy = (action - predictions).pow(2).sum(1).sqrt().mean().item()
            idm_accumulated_accuracy.append(accuracy)

        self.idm.eval()

        for batch in expert_dataset:
            state, _, next_state = batch
            state = state.to(self.device)
            next_state = next_state.to(self.device)

            with torch.no_grad():
                if self.discrete:
                    prediction = self.idm(torch.cat((state, next_state), dim=1))
                    # Compute probabilities to models logits
                    classes = np.arange(self.action_size)
                    prob = torch.nn.functional.softmax(prediction, dim=1).cpu().detach().numpy()
                    # Sample from the probabilities
                    samples = prob.cumsum(axis=1)
                    random = np.random.rand(prob.shape[1])
                    indexes = (samples < random).sum(axis=1)
                    action = classes[indexes]
                    # Convert to tensor
                    action = torch.tensor(action)
                    action = action.to(self.device)
                else:
                    action = self.idm(torch.cat((state, next_state), dim=1))

            self.optimizer_fn.zero_grad()
            predictions = self.forward(state)

            loss = self.loss_fn(predictions, action.squeeze().long())
            loss.backward()
            accumulated_loss.append(loss.item())
            self.optimizer_fn.step()

            accuracy: Number = None
            if self.discrete:
                accuracy = accuracy_fn(predictions, action.squeeze())
            else:
                accuracy = (action - predictions).pow(2).sum(1).sqrt().mean().item()
            accumulated_accuracy.append(accuracy)

        metrics = {
            "idm_loss": np.mean(idm_accumulated_loss),
            "idm_accuracy": np.mean(idm_accumulated_accuracy),
            "loss": np.mean(accumulated_loss),
            "accuracy": np.mean(accumulated_accuracy)
        }
        return metrics
