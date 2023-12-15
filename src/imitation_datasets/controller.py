"""Controller module for running experiments"""
from argparse import Namespace
import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
from os import listdir
from sys import platform

from torch.multiprocessing import set_start_method
from tqdm import tqdm
import psutil

from .experts import Experts
from .utils import CPUS, CollateFunction, Context, Experiment, EnjoyFunction


class Controller:
    """Controller for running experiments."""

    def __init__(
            self,
            enjoy: EnjoyFunction,
            collate: CollateFunction,
            amount: int,
            threads: int = 1,
            path: str = './dataset/'
    ) -> None:
        """Initialize the controller.

        Args:
            enjoy (EnjoyFunction): Function to run the expert.
            collate (CollateFunction): Function to collate the data.
            amount (int): Amount of episodes to run.
            threads (int, optional): Amount of threads to use. Defaults to 1.
            path (str, optional): Path to save the dataset. Defaults to './dataset/'.
        """
        self.enjoy = enjoy
        self.collate = collate
        self.threads = CPUS(threads)
        self.experiments = Experiment(amount)
        self.path = path

        self.pbar = None
        set_start_method('spawn', force=True)

    def create_folder(self, path: str) -> None:
        """Create a folder if it does not exist.

        Args:
            path (str): Path to the folder.
        """
        if not os.path.exists(path):
            os.makedirs(path)

    async def set_cpu(self, cpu: int) -> None:
        """Set the cpu affinity for the current process.

        Args:
            cpu (int): CPU index to use.
        """
        try:
            proc = psutil.Process()
            proc.cpu_affinity([int(cpu)])
            if 'linux' in platform:
                os.sched_setaffinity(proc.pid, [int(cpu)])
        except OSError:
            pass

    def enjoy_closure(self, opt: Namespace) -> EnjoyFunction:
        """Create a closure for the enjoy function.

        Args:
            opt (Namespace): Namespace with the arguments.

        Returns:
            EnjoyFunction: Enjoy function with part of the arguments.
        """
        os.system("set LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so")
        os.system("set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia")
        return partial(self.enjoy, expert=Experts.get_expert(opt.game))

    def collate_closure(self, opt: Namespace) -> CollateFunction:
        """Create a closure for the collate function.

        Args:
            opt (Namespace): Namespace with the arguments.

        Returns:
            CollateFunction: Collate function with part of the arguments.
        """
        path = f'{self.path}{opt.game}/'
        files = list(listdir(path))
        return partial(self.collate, data=files, path=path)

    async def enjoy_sequence(self, future: EnjoyFunction, executor: ProcessPoolExecutor) -> bool:
        """_summary_

        Args:
            future (EnjoyFunction): Enjoy function already with async future.
            executor (ProcessPoolExecutor): Executor to run the future.

        Returns:
            bool: Result of the future.
                  True if the expert was able to solve the game. False otherwise.
        """
        # Pre
        cpu = await self.threads.cpu_allock()
        await self.experiments.start()
        await self.set_cpu(cpu)

        # Enjoy
        result = await asyncio.get_event_loop().run_in_executor(executor, future)

        # Post
        self.threads.cpu_release(cpu)
        await self.experiments.stop(result)
        self.pbar.update(1 if result else 0)

        return result if result else await asyncio.gather(self.enjoy_sequence(future, executor))

    async def run(self, opt) -> None:
        """Run the experiments.

        Args:
            opt (Namespace): Namespace with the arguments.
        """
        path = f'{self.path}{opt.game}/'
        self.create_folder(path)

        tasks = []
        with ProcessPoolExecutor() as executor:
            for idx in range(self.experiments.amount):
                enjoy = self.enjoy_closure(opt)
                enjoy = partial(enjoy, path=path, context=Context(self.experiments, idx))
                task = asyncio.ensure_future(
                    self.enjoy_sequence(
                        enjoy,
                        executor
                    )
                )
                tasks.append(task)
            await asyncio.gather(*tasks)

    def start(self, opt: Namespace):
        """Start the experiments.

        Args:
            opt (Namespace): Namespace with the arguments.

        Raises:
            exception: Exception (general) raised during the execution.
        """
        try:
            if opt.mode in ['all', 'play']:
                self.pbar = tqdm(range(self.experiments.amount), desc='Running episodes')
                asyncio.run(self.run(opt))

            if opt.mode in ['all', 'collate']:
                self.pbar = tqdm(range(self.experiments.amount), desc='Running collate')
                collate = self.collate_closure(opt)
                collate()
        except Exception as exception:
            self.experiments.add_log(-99, exception)
            raise exception
        finally:
            self.experiments.write_log()
