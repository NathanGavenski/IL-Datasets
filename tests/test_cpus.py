import pytest
import multiprocessing


def setUp():
    from imitation_datasets.utils import CPUS
    return CPUS(4)


def tearDown(cpus) -> None:
    from imitation_datasets.utils import CPUS
    for cpu, status in cpus.cpus.items():
        if status:
            cpus.cpu_release(cpu)

    del CPUS


def test_cpu_init() -> None:
    from imitation_datasets.utils import CPUS
    try:
        cpus = CPUS(50)
        number_of_cpus = multiprocessing.cpu_count() - 1
        assert cpus.available_cpus == number_of_cpus
    finally:
        tearDown(cpus)


@pytest.mark.asyncio
async def test_cpu_allock() -> None:
    cpus = setUp()
    try:
        result = await cpus.cpu_allock()
        assert isinstance(result, int)
        assert cpus.cpus[result]
        assert len(cpus.cpus.keys()) == 1
        assert cpus.cpu_semaphore._value == 3

        result = await cpus.cpu_allock()
        assert len(cpus.cpus.keys()) == 2
        assert cpus.cpu_semaphore._value == 2
    finally:
        tearDown(cpus)


@pytest.mark.asyncio
async def test_cpu_release():
    cpus = setUp()

    try:
        result = await cpus.cpu_allock()
        cpus.cpu_release(result)
        assert not cpus.cpus[result]
        assert cpus.cpu_semaphore._value == 4
        assert cpus.cpu_release(result) is None
    finally:
        tearDown(cpus)
