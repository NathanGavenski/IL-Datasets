import asyncio
from typing import Any
from imitation_datasets.args import get_args
from imitation_datasets.controller import Controller
from imitation_datasets.experts import Policy
from imitation_datasets.utils import Context     

if __name__ == '__main__':
    opt = get_args()

    async def run(path: str, experiment: Context, expert: Policy) -> bool:
        await asyncio.sleep(1)
        return True

    def collate(path: str, data: list[str]) -> None:
        pass 
    
    x = Controller(run, 100, 1)
    x.start(opt)
