from argparse import Namespace
import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
from os import listdir
import psutil
from sys import platform

from torch.multiprocessing import set_start_method
from tqdm import tqdm

from .experts import Experts 
from .utils import CPUS, CollateFunction, Context, Experiment, EnjoyFunction


class Controller:
    def __init__(self, enjoy: EnjoyFunction, collate: CollateFunction, amount: int, threads: int = 1, path: str = './dataset/') -> None:
        self.enjoy = enjoy
        self.collate = collate
        self.threads = CPUS(threads)
        self.experiments = Experiment(amount)
        self.path = path

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

    def enjoy_closure(self, opt: Namespace) -> EnjoyFunction:
        os.system("set LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so")
        os.system("set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia")
        return partial(self.enjoy, expert=Experts.get_expert(opt.game))

    def collate_closure(self, opt: Namespace):
        path = f'{self.path}{opt.game}/'
        files = [f for f in listdir(path)]
        return partial(self.collate, data=files, path=path)

    async def enjoy_sequence(self, future: EnjoyFunction, executor: ProcessPoolExecutor) -> bool:
        # Pre
        cpu = await self.threads.cpu_allock()
        await self.experiments.start()
        await self.set_cpu(cpu)

        # Enjoy
        result = await asyncio.get_event_loop().run_in_executor(executor, future)

        # Post
        self.threads.cpu_release(cpu)
        await self.experiments.stop(result)
        self.pbar.update(1)

        return result

    async def run(self, opt) -> None:
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


    def start(self, opt):
        if opt.mode in ['all', 'play']: 
            self.pbar = tqdm(range(self.experiments.amount), desc='Running episodes')
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.run(opt))
            loop.close()

        if opt.mode in ['all', 'collate']:
            self.pbar = tqdm(range(self.experiments.amount), desc='Running collate')
            collate = self.collate_closure(opt)
            collate()
        
        self.experiments.write_log()
