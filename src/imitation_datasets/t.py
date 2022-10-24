import asyncio
from functools import partial
from multiprocessing import Process


async def run():
    print('oi')
    await asyncio.sleep(5)
    print('tchau')


if __name__ == "__main__":

    def run_closure(fn):
        loop = asyncio.new_event_loop()
        loop.run_until_complete(fn())
        loop.close()

    p = Process(target=run_closure, args=(task,))
    p.start()