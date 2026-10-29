# Inverse soft-Q Learning

Online implementation of IQ-Learn ([Garg et al., 2021](https://arxiv.org/abs/2106.12142)) for **continuous** action spaces, following the official MuJoCo experiments ([Div99/IQ-Learn](https://github.com/Div99/IQ-Learn)).
For discrete action spaces, use [Offline Inverse soft-Q Learning](offline_iqlearn.md).

A SAC agent (tanh-Gaussian actor and a soft Q critic with a target network) interacts with the environment.
At every step, the critic is trained with the IQ loss on a batch of expert and agent transitions (with $\chi^2$ regularization), followed by a SAC actor update.
Only real terminations count as `done`; episodes cut by the time limit are bootstrapped.

Each training epoch is `steps_per_epoch` (5000) environment steps.
Every `enjoy_criteria` epochs the policy is evaluated for `eval_episodes` (10) episodes on a separate environment, and the best checkpoint is kept.
Hyperparameters are loaded from [iqlearn.yaml](https://github.com/NathanGavenski/IL-Datasets/blob/main/src/imitation_datasets/baselines/methods/config/iqlearn.yaml) by environment name, falling back to `Default`.
Following the official scripts, Hopper uses the `v0` loss and Ant uses a temperature of `0.001`.

## Usage

```python
import gymnasium as gym
from imitation_datasets.baselines.methods import IQLearn
from imitation_datasets.baselines.benchmark import create_dataloader

dataloader = create_dataloader("NathanGavenski/Hopper-v4")
# 100 epochs * 5000 steps = 500k environment steps, evaluating every 5000 steps
policy = IQLearn(gym.make("Hopper-v4"), enjoy_criteria=1, verbose=True)
policy.train(100, train_dataset=dataloader).load()
```

::: imitation_datasets.baselines.methods.iqlearn.IQLearn
