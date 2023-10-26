import shutil

import gymnasium as gym
from torch.utils.data import DataLoader

from src.imitation_datasets.functions import baseline_enjoy, baseline_collate
from src.imitation_datasets.controller import Controller
from src.imitation_datasets.args import get_args
from src.imitation_datasets.dataset import BaselineDataset
from src.benchmark.methods.bc import BC
from src.imitation_datasets.dataset.metrics import performance


if __name__ == "__main__":
    shutil.rmtree("./dataset/")

    args = get_args()
    controller = Controller(baseline_enjoy, baseline_collate, args.episodes, args.threads)
    controller.start(args)

    print("Creating Dataset")
    env = gym.make("CartPole-v1")
    dataset = BaselineDataset("./dataset/cartpole/teacher.npz")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print("Creating BC")
    bc = BC(env, verbose=True)
    bc.train(100, dataloader)
    bc.load()
    aer = bc._enjoy()
    print("Model results:")
    print(f"\tAER: {aer}")
    print(f"\tPerformance: {performance(aer, 500, 9.8)}")
