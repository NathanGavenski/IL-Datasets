# Benchmarking

All benchmark results are available in the [benchmark_results.md](https://github.com/NathanGavenski/IL-Datasets/blob/main/benchmark_results.md).
The benchmark module requires the `benchmark` extras (see [Installation](install.md)).

## Methods

| Method | Class | Action space | Interacts with the environment |
|--------|-------|--------------|--------------------------------|
| [Behavioural Cloning](../algorithms/bc.md) | `BC` | Discrete and continuous | No |
| [Behavioural Cloning from Observation](../algorithms/bco.md) | `BCO` | Discrete and continuous | Yes |
| [Augmented Behavioural Cloning from Observation](../algorithms/abco.md) | `ABCO` | Discrete and continuous | Yes |
| [Imitating Unknown Policies via Exploration](../algorithms/iupe.md) | `IUPE` | Discrete and continuous | Yes |
| [Generative Adversarial Imitation Learning](../algorithms/gail.md) | `GAIL` | Discrete and continuous | Yes |
| [Inverse soft-Q Learning](../algorithms/iqlearn.md) | `IQLearn` | Continuous | Yes |
| [Offline Inverse soft-Q Learning](../algorithms/offline_iqlearn.md) | `OfflineIQLearn` | Discrete | No |

Every method follows its original paper (or official implementation) and is not tuned per environment beyond what the authors report.
Any unavoidable difference is listed in the method's page.

## Environments

Environments are registered in [registers.py](https://github.com/NathanGavenski/IL-Datasets/blob/main/src/imitation_datasets/baselines/registers.py), each with its [HuggingFace dataset](https://huggingface.co/collections/NathanGavenski/imitation-learning-datasets-6542982072defaf65937432d) and the reward of a random agent:

* **Classic control**: `CartPole-v1`, `MountainCar-v0`, `Acrobot-v1` and `LunarLander-v2`.
* **MuJoCo**: `InvertedPendulum-v4`, `Swimmer-v4`, `HalfCheetah-v4`, `Hopper-v4` and `Ant-v4`.

The `-v4` MuJoCo environments require `gymnasium==0.29.1` and `mujoco<3` (e.g., `2.3.7`); `LunarLander-v2` also requires `gymnasium<1.0` and `box2d`.

## Metrics

Each result is the evaluation of the best checkpoint over 100 episodes:

* **AER**: average episodic reward (± standard deviation).
* **Performance**: reward normalized between the random agent (0) and the expert (1), $\frac{r - r_{random}}{r_{expert} - r_{random}}$.

The best checkpoint is selected during training every `enjoy_criteria` epochs.

## Running the benchmark

To run all benchmarks locally, use:
```{bash}
$ python -m imitation_datasets.baselines.benchmark
```

This command will run all methods listed in `benchmark_methods` (`registers.py`) with all registered environments, and write the results to `benchmark_results_<METHODS>.md`.
Be careful, it will take a long time!
If you wish to run a single method or run them in parallel, use:
```{bash}
$ python -m imitation_datasets.baselines.benchmark --methods <METHOD NAME>
```

For example:
```{bash}
$ python -m imitation_datasets.baselines.benchmark --methods bc,bco
```
will run Behavioural Cloning and Behavioural Cloning from Observation.

The command-line benchmark trains each method for 5000 epochs, evaluating every 100 epochs.
For the online methods, where an epoch is a number of environment steps, train them directly instead (see the usage examples in [GAIL](../algorithms/gail.md) and [Inverse soft-Q Learning](../algorithms/iqlearn.md)):

```python
import gymnasium as gym
from imitation_datasets.baselines.benchmark import create_dataloader
from imitation_datasets.baselines.methods import IQLearn
from imitation_datasets.baselines.registers import mujoco

info = mujoco["Hopper-v4"]
dataloader = create_dataloader(info["path"])
metrics = IQLearn(gym.make("Hopper-v4"), enjoy_criteria=1) \
    .train(100, train_dataset=dataloader) \
    .load() \
    ._enjoy(teacher_reward=dataloader.dataset.average_reward, random_reward=info["random_reward"])
```

## Logs and checkpoints

Training curves are written to TensorBoard under `./benchmark_results/<method>/<environment>/` and checkpoints under `./tmp/<method>/<environment>/`.
To follow training, run:
```{bash}
$ tensorboard --logdir benchmark_results
```
