from argparse import ArgumentParser
import asyncio
from functools import partial
import os
import psutil
from sys import platform

from torch.multiprocessing import set_start_method
from tqdm import tqdm

from utils.experts import Experts 

from .utils import CPUS, Context, Experiment, EnjoyFunction

import warnings
import ctypes

warnings.filterwarnings("ignore")


class Controller:
    def __init__(self, func: EnjoyFunction, amount: int, threads: int = 1) -> None:
        self.enjoy = func
        self.threads = CPUS(threads)
        self.experiments = Experiment(amount)

        self.pbar = tqdm(range(self.experiments.amount), position=0, leave=True)
        set_start_method('spawn', force=True)

    def create_folder(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)

    async def set_cpu(self, cpu: int) -> None:
        proc = psutil.Process()
        proc.cpu_affinity([int(cpu)])
        if 'linux' in platform:
            os.sched_setaffinity(proc.pid, [int(cpu)])

    def enjoy_closure(self, opt: ArgumentParser) -> EnjoyFunction:
        os.system("set LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so")
        os.system("set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia")
        return partial(self.enjoy, expert=Experts.get_expert(opt.game))

    async def enjoy_sequence(self, future: EnjoyFunction) -> bool:
        # Pre
        cpu = await self.threads.cpu_allock()
        await self.experiments.start()
        await self.set_cpu(cpu)

        # Enjoy
        result = await future

        # Post
        self.threads.cpu_release(cpu)
        await self.experiments.stop(result)
        self.pbar.update(1)

        return result

    async def run(self, opt) -> None:
        path = f'./dataset/{opt.game}/'
        self.create_folder(path)

        tasks = []
        for idx in range(self.experiments.amount):
            enjoy = self.enjoy_closure(opt)
            task = asyncio.ensure_future(
                self.enjoy_sequence(
                    enjoy( 
                        path,
                        Context(self.experiments, idx),
                    )
                )
            )
            tasks.append(task)
        await asyncio.wait(tasks)


    def start(self, opt):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.run(opt))
        loop.close()
