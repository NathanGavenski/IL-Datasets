"""Module for benchmarking."""
import logging
from numbers import Number
from typing import List

import gymnasium as gym
from gymnasium.error import VersionNotFound, NameNotFound
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader

from .registers import benchmark_environments
from .registers import get_methods
from .args import get_args
from .methods.method import Method, Metrics
from imitation_datasets.dataset import BaselineDataset


def create_dataloader(path: str) -> DataLoader:
    """Creates dataloader based on the BaselineDataset.

    Args:
        path(str): HuggingFace path to dataset.

    Returns:
        dataloader (DataLoader): dataloader to use for training.
    """
    dataset = BaselineDataset(path, source="huggingface")
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)
    return dataloader


def benchmark_method(
    method: Method,
    environment: gym.Env,
    dataloader: DataLoader,
    teacher_reward: Number,
    random_reward: Number
) -> Metrics:
    """Function for training a method and evaluating.

    Args:
        method (Method): class for a method.
        environment (Env): environment to train the method.
        dataloader (DataLoader): dataloader to train the method.
        teacher_reward (Number): teacher reward to compute performance.
        random_reward (Number): random reward to compute performance.

    Returns:
        metrics (Metrics): resulting metrics for best checkpoint.
            aer (Dict[str, str]): average episodic reward.
            performance (Dict[str, str]) performance.
    """
    policy: Method = method(environment, verbose=True, enjoy_criteria=100)
    metrics = policy.train(5000, train_dataset=dataloader) \
        .load() \
        ._enjoy(teacher_reward=teacher_reward, random_reward=random_reward)
    aer = f"{round(metrics['aer'], 4)} ± {round(metrics['aer_std'], 4)}"
    performance = f"{round(metrics['performance'], 4)} ± {round(metrics['performance_std'], 4)}"
    return {"aer": aer, "performance": performance}


# pylint: disable=W0718
def benchmark(benchmark_methods: List[Method]) -> None:
    """Benchmark for all methods and environments listed on registers.py

    Args:
        benchmark_methods (List[Method]): list of benchmark methods to run.
    """
    file_name = "benchmark_results_"
    file_name += "-".join([str(method.__name__) for method in benchmark_methods])

    benchmark_results = []
    for environments in tqdm(benchmark_environments, desc="Benchmark Environments"):
        for name, info in environments.items():
            path, random_reward = info.values()

            try:
                environment = gym.make(name)
            except VersionNotFound:
                logging.error("benchmark: Version for environment does not exist")
            except NameNotFound:
                logging.error("benchmark: Environment name does not exist")
            except Exception:
                logging.error("benchmark: Generic error raised, probably dependency related")

            try:
                dataloader = create_dataloader(path)
            except FileNotFoundError:
                logging.error("benchmark: HuggingFace path is not valid")
                continue

            for method in tqdm(benchmark_methods, desc=f"Methods for environment: {name}"):
                try:
                    metrics = benchmark_method(
                        method,
                        environment,
                        dataloader,
                        dataloader.dataset.average_reward,
                        random_reward
                    )
                except Exception as exception:
                    logging.error(
                        "benchmark: Method %s did raise an exception during training",
                        method.__method_name__
                    )
                    logging.error(exception)
                    continue

                benchmark_results.append([
                    name,
                    method.__method_name__,
                    *metrics.values()
                ])

        table = tabulate(
            benchmark_results,
            headers=["Environment", "Method", "AER", "Performance"],
            tablefmt="github"
        )

        with open(f"./{file_name}.md", "w", encoding="utf-8") as _file:
            _file.write(table)


if __name__ == "__main__":
    args = get_args()
    methods = args.methods.split(",")
    benchmark(get_methods(methods))
