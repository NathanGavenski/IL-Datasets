"""Module for utility functions and classes used in the imitation_datasets package."""
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
import multiprocessing
import os
from typing import Any, Callable, DefaultDict, Union, List, Dict, Tuple
from typing_extensions import Self

import gymnasium as gym

from .experts import Policy


class Singleton(type):
    """Singleton metaclass."""
    _instances = {}

    def __call__(cls, *args, **kwargs) -> Self:
        """Call method for Singleton metaclass.

        Returns:
            Self: Singleton instance.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass
class Experiment:
    """Experiment dataclass to keep track of the experiments."""

    amount: int
    path: str = './logs.txt'
    waiting: int = field(init=False, default_factory=int)
    logs: DefaultDict[int, list] = field(
        init=False, default_factory=lambda: defaultdict(list))
    experiment_semaphore: asyncio.Lock = field(
        init=False, default=asyncio.BoundedSemaphore(value=1))

    def __post_init__(self) -> None:
        """Write in log file that the dataset creation has started."""
        if os.path.exists(self.path):
            os.remove(self.path)

        if not os.path.exists(self.path):
            with open(self.path, 'w', encoding='utf8') as log_file:
                log_file.write('#### Starting dataset creation ####\n')

    def is_done(self) -> bool:
        """Check if the experiment is done.
        
        Returns:
            bool: True if the experiment is done, False otherwise.
        """
        return self.amount == 0

    async def start(self, amount: int = 1) -> Tuple[bool, int]:
        """Start an experiment.

        Args:
            amount (int, optional): How many experiments are left to run. Defaults to 1.

        Returns:
            status (bool): True if the experiment can be started, False otherwise.
            amount (int): How many experiments are left to run.
        """
        await self.experiment_semaphore.acquire()
        if self.amount > 0:
            self.waiting += amount
            self.amount -= amount
            self.experiment_semaphore.release()
            return True, self.amount

        self.experiment_semaphore.release()
        return False, -1

    async def stop(self, status: bool, amount: int = 1) -> None:
        """Stop an experiment.
        
        Args:
            status (bool): True if the experiment was successful, False otherwise.
            amount (int, optional): How many experiments are left to run. Defaults to 1.
        """
        await self.experiment_semaphore.acquire()
        self.amount += 0 if status else amount
        self.waiting -= amount
        self.experiment_semaphore.release()

    def add_log(self, experiment: int, log: str) -> None:
        """Add a log to the experiment.
        
        Args:
            experiment (int): Experiment index.
            log (str): Log to add.
        """
        self.logs[experiment].append(log)

    def write_log(self) -> None:
        """Write the logs in the log file."""
        with open('./logs.txt', 'a', encoding='utf8') as log_file:
            for idx, logs in self.logs.items():
                for log in logs:
                    log_file.write(f'\nExperiment {idx}: {log}')
                log_file.write('\n')


@dataclass
class Context:
    """Context dataclass to keep track of the context of the experiment."""
    experiments: Experiment
    index: int

    def add_log(self, log: str) -> None:
        """Add a log to the experiment."""
        self.experiments.add_log(self.index, log)


@dataclass
class CPUS(metaclass=Singleton):
    """CPUS dataclass to keep track of the available CPUs."""
    # FIXME if the user is not the admin, the class should not be invoked

    available_cpus: int = field(default_factory=multiprocessing.cpu_count())
    cpus: DefaultDict[int, bool] = field(
        init=False, default_factory=lambda: defaultdict(bool))
    cpu_semaphore: asyncio.Lock = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the cpu_semaphore."""
        if self.available_cpus > multiprocessing.cpu_count() - 1:
            self.available_cpus = multiprocessing.cpu_count() - 1
        self.cpu_semaphore = asyncio.BoundedSemaphore(
            value=self.available_cpus)

    async def cpu_allock(self) -> int:
        """Acquire a CPU.

        Returns:
            int: CPU index.
        """
        await self.cpu_semaphore.acquire()
        for idx in range(self.available_cpus):
            if not self.cpus[idx]:
                self.cpus[idx] = True
                return idx

    def cpu_release(self, cpu_idx: int) -> None:
        """Release a CPU.

        Args:
            cpu_idx (int): CPU index.
        """
        self.cpus[cpu_idx] = False
        self.cpu_semaphore.release()


EnjoyFunction = Callable[[Policy, str, Context], bool]
CollateFunction = Callable[[str, List[str]], None]


class GymWrapper:
    """
        Wrapper for gym environment. Since Gymnasium and Gym version 0.26 
        there are some environments that were working under Gym-v.0.21 stopped 
        working. This wrapper just makes sure that the output for the environment 
        will always work with the version the user wants.
    """
    # FIXME Implement all functions from a gym environment -- missing render and others
    # TODO Gym got rid of the seed function, it would be nice to have one

    def __init__(self, name: str, version: str = "newest") -> None:
        """
        Args:
            name: gym environment name
            version: ["newest", "older"] = refers to the compatibility version. 

        In this case, "newest" is 0.26 and "older" is 0.21.
        """
        if version not in ["newest", "older"]:
            raise ValueError("Version has to be :" + ["newest", "older"])

        self.env = gym.make(name)
        self.version = version

    def reset(self) -> Union[Tuple[List[float], Dict[str, Any]], List[float]]:
        """
        Resets the framework and return the appropriate return.
        """
        state = self.env.reset()
        if self.version == "newest":
            return state

        if len(state) > 1:
            return state[0]
        return state

    def step(
            self,
            action: Union[float, int]
        ) -> Union[
            Tuple[List[float], float, bool, bool, Dict[str, Any]],
            Tuple[List[float], float, bool, Dict[str, Any]]
    ]:
        """
        Perform an action in the environment and return the appropriate return 
        according to version.
        """
        gym_return = self.env.step(action)
        if self.version == "newest":
            return gym_return

        if len(gym_return) > 4:
            state, reward, terminated, truncated, info = gym_return
            return state, reward, terminated or truncated, info

        return gym_return

    def render(self):
        """
        Return the render for the environment.
        """
        return self.env.render()

    def close(self) -> None:
        """
        Close the environment.
        """
        self.env.close()
