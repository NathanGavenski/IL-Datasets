import argparse
from random import choice
from collections import defaultdict
from functools import partial
from typing import Callable, Any
from numbers import Number
import os
from os import listdir
from os.path import join, isfile
import shutil
import pickle

from datasets import load_dataset
from datasets import Dataset as HFDataset
import gymnasium as gym
import maze
from maze.file_utils import convert_from_file
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.imitation_datasets.experts import Policy
from src.imitation_datasets.dataset.metrics import accuracy

from src.benchmark.methods.bc import BC
from src.benchmark.methods.gail import GAIL
from src.benchmark.methods.dagger import DAgger
from src.benchmark.methods.bco import BCO
from src.benchmark.methods.sqil import SQIL
from src.benchmark.methods.iupe import IUPE

FILE_PATH = "/benchmarking/nathan/github/maze-gym/src/"

methods = {
    "bc": BC,
    "gail": GAIL,
    "dagger": DAgger,
    "bco": BCO,
    "sqil": SQIL,
    "iupe": IUPE
}

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Maze-v0")
    parser.add_argument(
        "--method",
        type=str,
        default="bc",
        choices=list(methods.keys()),
        help="Method to use for training"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10000,
        help="Number of episodes to train"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for training"
    )
    return parser.parse_args()

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

class BCOExpertDataset(Dataset):
    def __init__(
        self,
        dataset: HFDataset,
        custom_transforms: Callable[[torch.Tensor], torch.Tensor] = None
    ) -> None:
        super().__init__()
        dataset = dataset.to_pandas()  # obs, rewards, episode_starts, info
        observations = dataset["obs"].to_list()
        actions = dataset["actions"].to_list()

        episode_starts = dataset["episode_starts"]
        starts = np.where(episode_starts == True)[0].tolist()
        ends = starts[1:] + [len(dataset["episode_starts"])]

        self.states = []
        self.next_states = []
        self.actions = []
        for start, end in zip(starts, tqdm(ends, desc="BCO Dataset")):
            self.states += observations[start:end - 1]
            self.next_states += observations[start + 1:end]
            self.actions += actions[start:end - 1]
        self.actions = torch.tensor(self.actions)

        self.transforms = custom_transforms
        self.device = "cpu" if not torch.cuda.is_available() else "cuda"

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index):
        obs = self.states[index]
        if self.transforms is not None:
            obs = self.transforms(Image.open(obs)).to(self.device)
        else:
            obs = torch.from_numpy(obs).to(self.device)

        next_obs = self.next_states[index]
        if self.transforms is not None:
            next_obs = self.transforms(Image.open(next_obs)).to(self.device)
        else:
            obs = torch.from_numpy(next_obs).to(self.device)

        action = self.actions[index].to(self.device)[None]
        return obs, action, next_obs

class RandomDataset(Dataset):
    def __init__(
        self,
        dataset: dict[str, Any],
        custom_transforms: Callable[[torch.Tensor], torch.Tensor] = None
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.transforms = custom_transforms
        if self.transforms is None:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
            ])
        self.device = "cpu" if not torch.cuda.is_available() else "cuda"

    def __len__(self) -> int:
        return len(self.dataset["obs"])

    def __getitem__(self, index) -> tuple[torch.Tensor]:
        obs = self.dataset["obs"][index]
        if self.transforms is not None:
            obs = self.transforms(Image.open(obs)).to(self.device)
        else:
            obs = torch.from_numpy(obs).to(self.device)

        next_obs = self.dataset["next_obs"][index]
        if self.transforms is not None:
            next_obs = self.transforms(Image.open(next_obs)).to(self.device)
        else:
            next_obs = torch.from_numpy(next_obs).to(self.device)

        action = torch.tensor(self.dataset["action"][index]).to(self.device)[None]
        return obs, action, next_obs

def maze_function(
    self,
    policy: Policy,
    transforms: Callable[[torch.Tensor], torch.Tensor] = None,
    **kwargs: dict[str, Any]
) -> dict[str, Number]:
    import maze
    from maze.file_utils import convert_from_file

    early_stop = kwargs.get("early_stop", True)
    return_ipos = kwargs.get("return_ipos", False)
    folder = kwargs.get("folder", "test")

    test_maze_path = f"{FILE_PATH}/environment/mazes/mazes5/{folder}/"
    mazes = [f for f in listdir(test_maze_path) if isfile(join(test_maze_path, f))]

    idm_save_path = f"./tmp/maze/idm/"
    os.makedirs(idm_save_path, exist_ok=True)
    idm_counter = len(listdir(idm_save_path)) + 1

    aer = []
    success_rate = []
    if return_ipos:
        i_pos = defaultdict(list)
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
            next_obs, reward, done, truncated, info = self.env.step(action)
            acc_reward += reward

            if return_ipos:
                Image.fromarray(obs).save(f"{idm_save_path}/{idm_counter}.png")
                Image.fromarray(next_obs).save(f"{idm_save_path}/{idm_counter}_next.png")
                i_pos["states"].append(f"{idm_save_path}/{idm_counter}.png")
                i_pos["actions"].append(action)
                i_pos["next_states"].append(f"{idm_save_path}/{idm_counter}_next.png")

            if already_visited_states[hash(next_obs.tobytes())] == 5 and early_stop:
                truncated = True
                acc_reward = -4

        aer.append(acc_reward)
        success_rate.append(done)

    if return_ipos:
        i_pos = {key: np.array(value) for key, value in i_pos.items()}

    metrics = {
        "aer": np.mean(aer),
        "aer_std": np.std(aer),
        "success_rate": np.mean(success_rate)
    }

    if return_ipos:
        return metrics, i_pos
    return metrics

def eval_model(
    self,
    dataset: DataLoader[TeacherDataset],
) -> dict[str, Number]:
    avg_acc = []
    self.policy.eval()

    for obs, actions, _ in tqdm(dataset, desc="Eval"):
        obs = obs.to(self.device)
        actions = actions.to(self.device)

        with torch.no_grad():
            pred = self.policy.forward(obs)
        avg_acc.append(accuracy(pred, actions.squeeze(1)))

    self.policy.train()
    return {
        "aer": np.mean(avg_acc),
        "accuracy": np.mean(avg_acc),
        "accuracy_std": np.std(avg_acc)
    }

def state_to_action(source: int, target: int, shape: tuple[int, int]) -> int:
    """Convert global index states into action (UP, DOWN, LEFT and RIGHT).

    Args:
        source (int): global index for the source in the maze.
        target (int): global index for the target in the maze.
        shape (Tuple[int, int]): maze shape.

    Returns:
        int: action to take.
    """
    width, _ = shape

    # Test left or right
    if source // width == target // width:
        if target > source:
            return 1
        return 3

    if target > source:
        return 0
    return 2

def create_random_dataset(environment) -> None:
    new_data = defaultdict(list)
    save_path = "./tmp/maze/random/"
    os.makedirs(save_path, exist_ok=True)

    random_counter = len(listdir(save_path)) + 1
    maze_structures = listdir(f"{FILE_PATH}/environment/mazes/mazes5/train/")
    for maze_structure in tqdm(maze_structures, desc="Maze Random Generation"):
        environment.load(*convert_from_file(join(f"{FILE_PATH}/environment/mazes/mazes5/train/", maze_structure)))
        obs, _ = environment.reset(options={"agent": True})
        done, truncated = False, False
        for _ in range(100):
            action = environment.action_space.sample()
            next_obs, reward, done, truncated, info = environment.step(action)

            Image.fromarray(obs).save(f"{save_path}/{random_counter}.png")
            Image.fromarray(next_obs).save(f"{save_path}/{random_counter}_next.png")

            new_data["obs"].append(f"{save_path}/{random_counter}.png")
            new_data["next_obs"].append(f"{save_path}/{random_counter}_next.png")
            new_data["action"].append(action)

            random_counter += 1
            obs = next_obs

            if done or truncated:
                obs, _ = environment.reset(options={"agent": True})

    with open(f"{save_path}/random_dataset.pkl", "wb") as f:
        pickle.dump(new_data, f)
    return new_data

def _collect_data(self) -> None:
    environment = self.environment
    new_data = defaultdict(list)

    save_path = "./tmp/maze/rollouts/"
    os.makedirs(save_path, exist_ok=True)
    image_counter = len(listdir(save_path)) + 1

    maze_structures = listdir(f"{FILE_PATH}/environment/mazes/mazes5/train/")
    for _ in range(self.hyperparameters.get("n_rollouts", 10)):
        maze_structure = choice(maze_structures)
        environment.load(*convert_from_file(join(f"{FILE_PATH}/environment/mazes/mazes5/train/", maze_structure)))
        obs, _ = environment.reset(options={"agent": True})
        solution = environment.solve("shortest")[0]
        done, truncated = False, False
        count = 0
        while not done and not truncated:
            action = state_to_action(solution[count], solution[count + 1], (5, 5))
            next_obs, reward, done, truncated, info = environment.step(action)
            
            Image.fromarray(obs).save(f"{save_path}/{image_counter}.png")
            new_data["obs"].append(f"{save_path}/{image_counter}.png")
            new_data["action"].append(action)
            
            count += 1
            image_counter += 1
            obs = next_obs
    
    new_data["action"] = torch.from_numpy(np.array(new_data["action"]))

    self.dataset.obs += new_data["obs"]
    self.dataset.actions = torch.cat((self.dataset.actions, new_data["action"]), dim=0)


if __name__ == "__main__":
    shutil.rmtree("./tmp/maze/rollouts/", ignore_errors=True)

    args = get_args()
    env = gym.make("Maze-v0", screen_width=64, screen_height=64, shape=(5, 5), render_mode="rgb_array")
    custom_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize(64)])

    Method = methods[args.method]
    method = Method(env, enjoy_criteria=100, verbose=True)
    env = gym.make("Maze-v0", screen_width=600, screen_height=600, shape=(5, 5), render_mode="rgb_array")
    method.environment = env

    if args.method == "dagger":
        _collect_data = partial(_collect_data, self=method)
        setattr(method, "_collect_data", _collect_data)


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
        batch_size=args.batch_size,
        shuffle=True,
    )
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    online = False
    if online:
        maze_function = partial(
            maze_function, self=method,
            policy=method, folder="eval",
            transforms=custom_transforms
        )
        setattr(method, "_enjoy", maze_function)
    else:
        if args.method in ["bco", "iupe"]:
            maze_function = partial(
                maze_function, self=method,
                policy=method, folder="train",
                transforms=custom_transforms,
                early_stop=False,
            )
            setattr(method, "_enjoy", maze_function)
        else:
            eval_model = partial(eval_model, self=method, dataset=eval_dataloader)
            setattr(method, "_enjoy", eval_model)

    epochs = 5000
    if args.method == "dagger":
        method.train(epochs, train_dataloader, None, eval_dataloader)
    if args.method in ["bco", "iupe"]:

        if os.path.exists("./tmp/maze/random/random_dataset.pkl"):
            with open("./tmp/maze/random/random_dataset.pkl", "rb") as f:
                dataset = pickle.load(f)
        elif os.path.exists("./tmp/maze/random/"):
            shutil.rmtree("./tmp/maze/random/", ignore_errors=True)
            dataset = create_random_dataset(env)
        else:
            dataset = create_random_dataset(env)

        idm_dataset = RandomDataset(dataset, custom_transforms=custom_transforms)
        idm_dataloader = DataLoader(
            dataset=idm_dataset,
            batch_size=args.batch_size,
            shuffle=True,
        )

        train_dataset = BCOExpertDataset(
            load_dataset("NathanGavenski/Maze-v0_5x5", split="train"),
            custom_transforms=custom_transforms
        )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
        )

        eval_dataset = BCOExpertDataset(
            load_dataset("NathanGavenski/Maze-v0_5x5", split="validation"),
            custom_transforms=custom_transforms
        )
        eval_dataloader = DataLoader(
            dataset=eval_dataset,
            batch_size=args.batch_size,
            shuffle=True,
        )

        train = {
            "expert_dataset": train_dataloader,
            "idm_dataset": idm_dataloader,
        }
        eval = {
            "expert_dataset": eval_dataloader,
            "idm_dataset": None,
        }

        method.train(epochs, train, eval)
    else:
        method.train(epochs, train_dataloader, eval_dataloader)
    
