# Available Datasets

The IL-Datasets also come with a default PyTorch dataset, called `BaselineDataset`.
It uses the pattern set by the `baseline_collate` function, and it allows the use of HuggingFace datasets created by the `baseline_to_huggingface` function.
The dataset list for benchmarking is under development, so to check all new versions, you can visit our collection on [HuggingFace](https://huggingface.co/collections/NathanGavenski/imitation-learning-datasets-6542982072defaf65937432d).

## BaselineDataset
To use the Baseline dataset, you can use a **local file**:
```python
from src.imitation_datasets.dataset import BaselineDataset
BaselineDataset(f"./dataset/cartpole/teacher.npz")
```

Or a **HuggingFace** path:
```python
from src.imitation_datasets.dataset import BaselineDataset
BaselineDataset(f"NathanGavenski/CartPole-v1", source="huggingface")
```

## Train and Evaluation splits

BaselineDataset allows for fewer episodes and splitting for `evaluation` and `train`.

```python
from src.imitation_datasets.dataset import BaselineDataset
dataset_train = BaselineDataset(f"NathanGavenski/CartPole-v1", source="huggingface", n_episodes=100)
dataset_eval = BaselineDataset(f"NathanGavenski/CartPole-v1", source="huggingface", n_episodes=100, split="eval")
```
