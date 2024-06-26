# Installing IL-Datasets

IL-Datasets install everythings it depends on, with the exception of PyTorch and Gym/Gymnasium.
We do this so we would not overide the current installed versions you use on yout local environment.
Therefore, before running IL-Datasets don't forget to install the versions you need!

## Requirements
The project supports Python versions `3.8`~`3.11`, the latest PyTorch and Gymnasium versions, and `v0.21.0` of Gym.


## PyPi

IL-Datasets is available on PyPi:

```{bash}
# Stable version
pip install il-datasets
```

if you plan to use the benchmark module, please use:
```{bash}
# Stable version
pip install "il-datasets[benchmark]"
```

## Local

But if you prefer, you can install it from the source:
```{bash}
git clone https://github.com/NathanGavenski/IL-Datasets.git
cd IL-Datasets
pip install -e .
```

## Docker image

If you want to run IL-Datasets with a docker to test, this project has a `Dockerfile`.
Currently, the files is configures for the AAMAS demonstration, which means that it instantiates the notebooks created to exemplify each part of the package (data creation, training assistance and benchmarking).

To build and run the docker image:
```bash
docker build -t ildatasets:latest .
docker run -p 127.0.0.1:8888:8888 ildatasets:latest
```

### Local Requirements

All requirements for the `imitation_datasets` module are listed in [requirements.txt](https://github.com/NathanGavenski/IL-Datasets/blob/main/requirements/requirements.txt).
These requirements are required by the module and are installed together with the `IL-Datasets`.
For requirements to use the `benchmark` module, use both the `imitation_datasets` requirements and the ones listed in [benchmark.txt](https://github.com/NathanGavenski/IL-Datasets/blob/main/requirements/benchmark.txt).
Development requirements are listed at [dev.txt](https://github.com/NathanGavenski/IL-Datasets/blob/main/requirements/dev.txt).
We do not recommend using these dependencies outside development.
They use an outdated version from gym `v0.21.0` to test the `GymWrapper` class.

