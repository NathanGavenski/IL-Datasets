# Generative Adversarial Imitation Learning

Implementation of GAIL ([Ho and Ermon, 2016](https://arxiv.org/abs/1606.03476)).
A discriminator learns to tell expert `(state, action)` pairs from the agent's pairs, and its output is used as the reward to train the policy.

Each training epoch collects `n_steps` environment steps, updates the discriminator against the expert data and then updates the policy.
Hyperparameters are loaded from [gail.yaml](https://github.com/NathanGavenski/IL-Datasets/blob/main/src/imitation_datasets/baselines/methods/config/gail.yaml) by environment name (e.g., `Hopper`), falling back to `Default`.

## Differences from the paper

* The generator is optimized with PPO ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)) instead of TRPO.
* The discriminator reward is configurable with the `reward` key:
    * `'gail'`: $-\log(1 - D(s, a))$, the reward from the paper. It is always positive, so it biases the policy towards long episodes, which fails in environments where ending the episode is the goal (e.g., MountainCar, Acrobot).
    * `'logit'`: $\log D(s, a) - \log(1 - D(s, a))$. It is negative while the discriminator wins, so it biases the policy towards short episodes, which fails in environments where termination means failure.

    `'logit'` is the default; InvertedPendulum, Hopper and Ant use `'gail'`.

## Usage

```python
import gymnasium as gym
from imitation_datasets.baselines.methods import GAIL
from imitation_datasets.baselines.benchmark import create_dataloader

dataloader = create_dataloader("NathanGavenski/Hopper-v4")
# 1465 epochs * 2048 steps ~= 3M environment steps
policy = GAIL(gym.make("Hopper-v4"), enjoy_criteria=10, verbose=True)
policy.train(1465, train_dataset=dataloader).load()
```

::: imitation_datasets.baselines.methods.gail.GAIL
