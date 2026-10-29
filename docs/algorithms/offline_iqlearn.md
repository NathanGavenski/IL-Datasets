# Offline Inverse soft-Q Learning

Offline implementation of IQ-Learn ([Garg et al., 2021](https://arxiv.org/abs/2106.12142)) for **discrete** action spaces.
The soft Q-function is learned only from the expert data, without interacting with the environment, and the policy is given by the Q-values: the greedy action when deterministic, otherwise sampled from a softmax with `temperature`.
For continuous action spaces, use the online [Inverse soft-Q Learning](iqlearn.md).

Hyperparameters are loaded from [offline_iqlearn.yaml](https://github.com/NathanGavenski/IL-Datasets/blob/main/src/imitation_datasets/baselines/methods/config/offline_iqlearn.yaml) by environment name, falling back to `Default`.

## Usage

```python
import gymnasium as gym
from imitation_datasets.baselines.methods import OfflineIQLearn
from imitation_datasets.baselines.benchmark import create_dataloader

dataloader = create_dataloader("NathanGavenski/Acrobot-v1")
policy = OfflineIQLearn(gym.make("Acrobot-v1"), enjoy_criteria=100, verbose=True)
policy.train(5000, train_dataset=dataloader).load()
```

::: imitation_datasets.baselines.methods.offline_iqlearn.OfflineIQLearn
