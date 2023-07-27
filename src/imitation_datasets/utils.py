import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
import multiprocessing
import os
from typing import Any, Callable, DefaultDict, Union, List, Dict, Tuple

import gymnasium as gym

from .experts import Policy


@dataclass
class Experiment:
    amount: int
    path: str = './logs.txt'
    waiting: int = field(init=False, default_factory=int)
    logs: DefaultDict[int, list] = field(init=False, default_factory=lambda: defaultdict(list))
    experiment_semaphore: asyncio.Lock = field(init=False, default=asyncio.BoundedSemaphore(value=1))

    def __post_init__(self) -> None:
        if not os.path.exists(self.path):
            with open(self.path, 'w') as f:
                f.write('#### Starting dataset creation ####\n')
        else:
            os.remove(self.path)

    def is_done(self) -> bool:
        return self.amount == 0

    async def start(self, amount: int = 1) -> Union[bool, int]:
        await self.experiment_semaphore.acquire()
        if self.amount > 0:
            self.waiting += amount
            self.amount -= amount
            self.experiment_semaphore.release()
            return True, self.amount
        else:
            self.experiment_semaphore.release()
            return False, -1

    async def stop(self, status: bool, amount: int = 1) -> None:
        await self.experiment_semaphore.acquire()
        self.amount += 0 if status else amount
        self.waiting -= amount
        self.experiment_semaphore.release()

    def add_log(self, experiment: int, log: str) -> None:
        self.logs[experiment].append(log)

    def write_log(self) -> None:
        with open('./logs.txt', 'a') as f:
            for idx, logs in self.logs.items():
                for log in logs:
                    f.write(f'\nExperiment {idx}: {log}')
                f.write('\n')


@dataclass
class Context:
    experiments: Experiment
    index: int

    def add_log(self, log: str) -> None:
        self.experiments.add_log(self.index, log)


@dataclass
class CPUS:
    available_cpus: int = field(default_factory=multiprocessing.cpu_count())
    cpus: DefaultDict[int, bool] = field(init=False, default_factory=lambda: defaultdict(bool))
    cpu_semaphore: asyncio.Lock = field(init=False)

    def __post_init__(self) -> None:
        if self.available_cpus > multiprocessing.cpu_count():
            self.available_cpus = multiprocessing.cpu_count()
        self.cpu_semaphore = asyncio.BoundedSemaphore(value=self.available_cpus)

    async def cpu_allock(self) -> int:
        await self.cpu_semaphore.acquire()
        for idx in range(self.available_cpus):
            if not self.cpus[idx]:
                self.cpus[idx] = True
                return idx
    
    def cpu_release(self, cpu_idx: int) -> None:
        self.cpus[cpu_idx] = False
        self.cpu_semaphore.release()

EnjoyFunction = Callable[[Policy, str, Context], bool]
CollateFunction = Callable[[str, list[str]], None]


# TODO create an actual wrapper that implements all functions (missing render and others)
# FIXME Gym got rid of the seed function, it would be nice to have one
class GymWrapper:
    """
        Wrapper for gym environment. Since Gymnasium and Gym version 0.26 
        there are some environments that were working under Gym-v.0.21 stopped 
        working. This wrapper just makes sure that the output for the environment 
        will always work with the version the user wants.
    """

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
        else:
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
