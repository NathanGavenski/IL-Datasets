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

    print("Creating Training Dataset")
    dataset_train = BaselineDataset("./dataset/cartpole/teacher.npz", n_episodes=700)
    dataloader_train = DataLoader(dataset_train, batch_size=2048, shuffle=True)

    print("Creating Evaluation Dataset")
    dataset_eval = BaselineDataset("./dataset/cartpole/teacher.npz", n_episodes=700, split="eval")
    dataloader_eval = DataLoader(dataset_eval, batch_size=2048, shuffle=True)

    print("Creating BC")
    env = gym.make("CartPole-v1")
    bc = BC(env, verbose=True, enjoy_criteria=10)
    bc.train(1000, train_dataset=dataloader_train, eval_dataset=dataloader_eval)
    bc.load()
    aer = bc._enjoy(render=True)["aer"]
    print("Model results:")
    print(f"\tAER: {aer}")
    print(f"\tPerformance: {performance(aer, 500, 9.8)}")
