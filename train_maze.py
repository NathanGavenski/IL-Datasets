from collections import defaultdict
from functools import partial
from typing import Callable, Any
from numbers import Number
from os import listdir
from os.path import join, isfile

from datasets import load_dataset
from datasets import Dataset as HFDataset
import gymnasium as gym
import maze
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.imitation_datasets.experts import Policy
from src.benchmark.methods.bc import BC

FILE_PATH = "/home/nathan/Documents/git/maze-gym/src/"


def maze_function(
    self,
    policy: Policy,
    transforms: Callable[[torch.Tensor], torch.Tensor] = None,
    **kwargs: dict[str, Any]
) -> dict[str, Number]:
    import maze
    from maze.file_utils import convert_from_file

    folder = kwargs.get("folder", "test")
    test_maze_path = f"{FILE_PATH}/environment/mazes/mazes5/{folder}/"
    mazes = [f for f in listdir(test_maze_path) if isfile(join(test_maze_path, f))]

    aer = []
    success_rate = []
    for maze_path in tqdm(mazes, desc=f"Maze {folder.capitalize()}"):
        self.env = gym.make(
            "Maze-v0", screen_width=600, screen_height=600, shape=(5, 5), render_mode="rgb_array"
        )
        structure, variables = convert_from_file(join(test_maze_path, maze_path))
        self.env.load(structure, variables)

        already_visited_states = defaultdict(int)
        obs, _ = self.env.reset(options={"agent": True})
        done, truncated = False, False
        acc_reward = 0
        while not done and not truncated:
            already_visited_states[hash(obs.tobytes())] += 1
            action = policy.predict(obs, transforms=transforms)
            obs, reward, done, truncated, info = self.env.step(action)
            acc_reward += reward

            if already_visited_states[hash(obs.tobytes())] == 5:
                truncated = True
                acc_reward = -4

        aer.append(acc_reward)
        success_rate.append(done)
    return {
        "aer": np.mean(aer),
        "aer_std": np.std(aer),
        "success_rate": np.mean(success_rate)
    }


class TeacherDataset(Dataset):

    def __init__(
        self,
        dataset: HFDataset,
        custom_transforms: Callable[[torch.Tensor], torch.Tensor] = None
    ) -> None:
        super().__init__()
        dataset = dataset.to_pandas()
        self.obs = dataset["obs"].to_list()
        if not isinstance(self.obs[0], str):
            torch.tensor(dataset["actions"])

        self.actions = torch.tensor(dataset["actions"].to_list())
        self.rewards = torch.tensor(dataset["rewards"].to_list())
        self.episode_starts = torch.tensor(dataset["episode_starts"].to_list())

        try:
            self.info = dataset["info"].to_list()
        except KeyError:
            self.info = []

        self.transforms = custom_transforms
        if self.transforms is None and isinstance(self.obs[0], str):
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
            ])
        self.device = "cpu" if not torch.cuda.is_available() else "cuda"

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, index) -> tuple[torch.Tensor]:
        obs = self.obs[index]
        if self.transforms is not None:
            obs = self.transforms(Image.open(obs)).to(self.device)
        else:
            obs = torch.from_numpy(obs).to(self.device)
        action = self.actions[index].to(self.device)[None]
        return obs, action, []


if __name__ == "__main__":
    env = gym.make("Maze-v0", screen_width=64, screen_height=64, shape=(5, 5))
    custom_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize(64)])

    method = BC(env, enjoy_criteria=100, verbose=True)
    maze_function = partial(
        maze_function, self=method,
        policy=method, folder="eval",
        transforms=custom_transforms
    )
    setattr(method, "_enjoy", maze_function)

    train_dataset = TeacherDataset(
        load_dataset("NathanGavenski/Maze-v0_5x5", split="train"),
        custom_transforms=custom_transforms
    )
    eval_dataset = TeacherDataset(
        load_dataset("NathanGavenski/Maze-v0_5x5", split="validation"),
        custom_transforms=custom_transforms
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=True,
    )
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=8,
        shuffle=True,
    )
    method.train(1000, train_dataloader, eval_dataloader)
    
