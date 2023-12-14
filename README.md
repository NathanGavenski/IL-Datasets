# IL Datasets

Hi, welcome to the Imitation Learning (IL) Datasets.
Something that always bothered me a lot was how difficult it was to find good weights for an expert, trying to create a dataset for different state-of-the-art methods, and also having to run all methods due to no common datasets.
For these reasons, I've created this repository in an effort to make it more accessible for researchers to create datasets using experts from the [Hugging Face](https://huggingface.co/models?pipeline_tag=reinforcement-learning).
IL-Datasets provides teacher weights for different environments, a multi-threading solution for creating datasets faster, datasets for a set of environments (check the bottom of this document to see which environments are already released), and a benchmark for common imitation learning methods.
We hope that by releasing these features, we can make the barrier to learning and researching imitation learning more accessible.

**This project is under development. If you are interested in helping, feel free to contact [me](https://nathangavenski.github.io/).**

## Requirements

The project supports Python versions `3.8`~`3.11`.
All requirements for the `imitation_datasets` package are listed in [requirements.txt](https://github.com/NathanGavenski/IL-Datasets/blob/main/requirements/requirements.txt). These requirements are required by the package and are installed together with the `IL-Datasets`.
For requirements to use the `benchmark` package, use both the `imitation_datasets` requirements and the ones listed in [benchmark.txt](https://github.com/NathanGavenski/IL-Datasets/blob/main/requirements/benchmark.txt).
Development requirements are listed at [dev.txt](https://github.com/NathanGavenski/IL-Datasets/blob/main/requirements/dev.txt). We do not recommend using these dependencies outside development. They use an outdated version from gym `v0.21.0` to test the `GymWrapper` class.

## Install

The package is available on PyPi:
```bash
# Stable version
pip install il-datasets
```

But if you prefer, you can install it from the source.
```bash
git clone https://github.com/NathanGavenski/IL-Datasets.git
cd IL-Datasets
pip install -e .
```

## How does it work?

This project also works with multithreading, which should accelerate the dataset creation.
It consists of one ``Controller`` class, which requires two different functions to work: (i) a ``enjoy`` function (for the agent to play and record an episode); and a (ii) ``collate`` function (for putting all episodes together).

---

The ``enjoy`` function will receive 3 parameters and return 1:
```python
"""
Args:
   path (str): where the episode is going to be recorded
   experiment (Context): A class for recording all information (if you don't want to use `print` - keeping the console clear)
   expert (Policy): A model based on the [StableBaselines3](https://stable-baselines3.readthedocs.io/en/master/) `BaseAlgorithm`.

Returns:
   status (bool): Whether it was successful or not
"""
```

Obs: To use the model you can call ``predict``, the policy class already has the correct form of using it (a.k.a., how the StableBaselines3 uses).

---

The ``collate`` function will receive 2 parameters and return 1:

```python
"""
Args:
   path (str): Where it should save the final dataset
   episodes  (list[str]): A list of paths for each file

Returns:
   status (bool): Whether it was successful or not
"""
```

## Default functions

The `imitation_datasets` package also comes with a set of default functions, so you don't need to implement a `enjoy` and a `collate` function in every project.
The resulting dataset will be a `NpzFile` with the following data:
```python
"""
Data:
   obs (list[list[float]): gym environment observation. Size [steps, observations space].
   actions (list[float]): agent action. Size [steps, action] (1 if single action, n if multiple actions).
   rewards (list[int]): reward from the action with the observations (e.g., r(obs, action)). Size [steps, ].
   episode_returns (list[float]): accumulated reward for each episode. Size [number of peisodes, ].
   episode_starts (list[bool]): whether the episode started at the current observation. Size [steps, ].
"""
```

A small functional example of how to use the given functions:
```python
# python <script> --game cartpole --threads 4 --episodes 1000 --mode all
from imitation_datasets.functions import baseline_enjoy, baseline_collate
from imitation_datasets.controller import Controller
from imitation_datasets.args import get_args

args = get_args()
controller = Controller(baseline_enjoy, baseline_collate, args.episodes, args.threads)
controller.start(args)
```

This script will use the pre-registered `CartPole-v1` environment with the HuggingFace weights and create a `teacher.npz` dataset file in `./dataset/cartpole/teacher.npz`.

## Registered environments

IL-Datasets comes with some already registered weights from HuggingFace.
To check which environments are already registered, check under the `src.imitation_datasets.registers` folder.

## Registering new experts

If you would like to add new experts locally, you can call the [Experts](./utils/experts.py) class. It uses the following structure:

```python
"""
Args:
   identifier (str): Name for calling the expert (e.g., cartpole).
   Policy (Policy): a dataclass with:
      name (str): Gym environment name
      repo_id (str): HuggingFace repo identification
      filename (str): HuggingFace weights file name
      threshold (float): How much reward should the episode accumulate to be considered good
      algo (BaseAlgorithm): The class from StableBaselines3
"""
```

If not using StableBaselines, you can load a `Policy` and not call the `load()` function (which downloads weights from HuggingFace).
Moreover, the expert has to have a `predict` function that receives:

```python
"""
Args:
   obs (Tensor): current environment state
   state (Tensor): Model's internal state
   deterministic (bool): if it should explore or not.
"""
```

## Datasets

The IL-Datasets also come with a default PyTorch dataset, called `BaselineDataset`. It uses the pattern set by the `baseline_collate` function, and it allows the use of HuggingFace datasets created by the `baseline_to_huggingface` function.
The dataset list for benchmarking is under development, so to check all new versions, you can visit our collection on [HuggingFace](https://huggingface.co/collections/NathanGavenski/imitation-learning-datasets-6542982072defaf65937432d).

To use the Baseline dataset, you can use a local file:
```python
from src.imitation_datasets.dataset import BaselineDataset
BaselineDataset(f"./dataset/cartpole/teacher.npz")
```

Or a HuggingFace path:
```python
from src.imitation_datasets.dataset import BaselineDataset
BaselineDataset(f"NathanGavenski/CartPole-v1", source="huggingface")
```

Finally, the dataset allows for fewer episodes and splitting for `evaluation` and `train`.
```python
from src.imitation_datasets.dataset import BaselineDataset
dataset_train = BaselineDataset(f"NathanGavenski/CartPole-v1", source="huggingface", n_episodes=100)
dataset_eval = BaselineDataset(f"NathanGavenski/CartPole-v1", source="huggingface", n_episodes=100, split="eval")
```

## Benchmark

Last but not least, IL-Datasets comes with its own benchmarking.
It uses IL methods on already published datasets to provide consistent results for researchers who also use our datasets.
Currently, we support:
   
| Algorithm | Implementation | Benchmark
| --- | --- | :---: |
| Behavioural Cloning | [`benchmark.methods.bc`](./src/benchmark/methods/bc.py) | ✅ |
| Behavioural Cloning from Observation | [`benchmark.methods.bco`](./src/benchmark/methods/bco.py) | ✅ |

However, our plan is to implement more state-of-the-art methods.

You can check the current benchmark results at [benchmark_results.md](https://github.com/NathanGavenski/IL-Datasets/blob/main/benchmark_results.md).

---
## This repository is under development

Here is a list of the upcoming releases:

- [ ] Create actual documentation
- [ ] Benchmark methods
   - [x] Behavioural Cloning
   - [x] Behavioural Cloning from Observation
   - [ ] Imitating Latent Policies from Observation
   - [ ] Augmented Behavioural Cloning from Observation
   - [ ] Imitating Unkown Policies via Exploration
   - [ ] Generative Adversarial Imitation Learning
   - [ ] Generative Adversarial Imitation Learning from Observation
   - [ ] Off-Policy Imitation Learning from Observations
   - [ ] Model-Based Imitation Learning From Observation Alone
- [ ] Benchmark environments
   - [x] CartPole-v1
   - [x] MountainCar-v0
   - [x] Acrobot-v1
   - [ ] LunarLander-v2
   - [ ] Ant-v3
   - [ ] Hopper-v3
   - [ ] HalfCheetah-v3
   - [ ] Walker-v3
   - [ ] Humanoid-v3
   - [ ] Swimmer-v3

 Although there are a lot of environments and methods to go through, this repository will be considered done once the documentation and the installation of the benchmarks are done. We don't have a plan for releases for environments and methods yet.

## If you like this repository, be sure to check my other projects:

### Development-based
- [An easy to use Wrapper for Tensorboard](https://github.com/NathanGavenski/Tensorboard-Wrapper)
- [A watcher for python to facilitate development of small projects](https://github.com/NathanGavenski/python-watcher)

### Academic
- [Self-Supervised Adversarial Imitation Learning (IJCNN)](https://arxiv.org/pdf/2304.10914.pdf)
- [How Resilient are Imitation Learning Methods to Sub-Optimal Experts? (BRACIS)](https://link.springer.com/chapter/10.1007/978-3-031-21689-3_32)
- [Self-supervised imitation learning from observation (MSc dissertation)](https://repositorio.pucrs.br/dspace/bitstream/10923/17536/1/000500266-Texto%2Bcompleto-0.pdf)
- [Imitating Unknown Policies via Exploration (BMVC)](https://arxiv.org/pdf/2008.05660.pdf)
- [Augmented behavioral cloning from observation (IJCNN)](https://arxiv.org/pdf/2004.13529.pdf)

