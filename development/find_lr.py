import sys
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder

import gymnasium as gym

from imitation_datasets.dataset import BaselineDataset
from benchmark.methods.bc import BC


class LrFinderDataset(Dataset):
    def __init__(self, dataset, return_cat=False):
        super().__init__()
        self.dataset = dataset
        self.return_cat = return_cat

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        s, a, nS = self.dataset[index]
        if self.return_cat:
            return torch.cat((s[None], nS[None]), axis=1).squeeze(), a
        else:
            return s, a.squeeze().long()


class LrFinderModel(BC):
    def forward(self, input):
        return self.policy(input).long()


if __name__ == "__main__":
    env_name = sys.argv[1]
    env = gym.make(env_name)
    bc = BC(env)

    train_dataset = BaselineDataset(
        f"NathanGavenski/{env_name}",
        source="huggingface",
        n_episodes=700,
        split="train"
    )
    eval_dataset = BaselineDataset(
        f"NathanGavenski/{env_name}",
        source="huggingface",
        n_episodes=700,
        split="eval"
    )

    train = LrFinderDataset(train_dataset)
    train = DataLoader(train, batch_size=2048, shuffle=True)
    eval = LrFinderDataset(eval_dataset)
    eval = DataLoader(eval, batch_size=2048, shuffle=True)

    model = LrFinderModel(env)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(bc.policy.parameters(), lr=1e-4)
    lr_finder = LRFinder(
        bc.policy,
        optimizer,
        criterion,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # The paper proposes using between 2 and 8 epochs
    iterations = len(train) * 8

    lr_finder.reset()
    lr_finder.range_test(
        train,
        val_loader=eval,
        end_lr=5e-3,
        num_iter=iterations,
        step_mode="linear"
    )

    (_, lrs), (_, losses) = lr_finder.history.items()
    min_grad_idx = (np.gradient(np.array(losses))).argmin()
    with open(f"./{env_name}.txt", "w") as _file:
        _file.write(f"{env_name}: {lrs[min_grad_idx]}")
