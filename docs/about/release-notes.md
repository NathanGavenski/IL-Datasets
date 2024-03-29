# Release Notes

## v0.4.0
### Benchmarking

Now IL-Datasets has its own benchmarking! We are adding new methods and environments to the repository.
For a full list of the methods and environments planned for release, please check the repository [readme.md](https://github.com/NathanGavenski/IL-Datasets/blob/main/README.md#this-repository-is-under-development) file.

### Support for benchmark requirements

We split the `imitation_datasets` and `benchmark` modules requirements.

```
pip install il-datasets
```

will only install requirements regarding the `imitation_datasets` module. For using `benchmark` please use:

```
pip install "il-datasets[benchmark]"
```

---

**Full Changelog**: https://github.com/NathanGavenski/IL-Datasets/compare/0.3.0...0.4.0


## v0.3.0

This version adds another point from the TODO list, **Datasets**!

Now, if you use the `baseline_enjoy` and `baseline_collate` functions, you can use the `BaselineDataset`.
The datasets will load the generated numpy file and organize all entries to be (s_t, a_t, s_{t+1}), provide the average reward for all episodes and also allow for fewer episodes with the parameter `n_episodes`.

Alongside the dataset, I've implemented a HuggingFace solution as well as utility functions that allow users to upload their datasets to the HuggingFace website. There is already an example at: https://huggingface.co/datasets/NathanGavenski/CartPole-v1
In the future, these datasets will be used for benchmarking, but for now, it allows for storing outside drivers (such as Google's and Microsoft's) 

This version also comes with some QoL improvements, such as pylint, and unit tests, so the code is more readable and also more stable.

Finally, with this release, I've implemented some metrics: `performance`, `average episodic reward` and `accuracy`.

### Future release sneak peek

It is my plan that the future release will introduce benchmarking to IL-Datasets.
With benchmarking, we will host a set of different datasets for common environments in the IL literature.
This should help all researchers (including myself) to stop running different methods for each experiment.

---

**Full Changelog**: https://github.com/NathanGavenski/IL-Datasets/compare/0.2.0...0.3.0

## V0.2.0

### New Features

- Added support for Gymnasium and Gym version 0.26.0. 
- Created template functions for `enjoy` and `collate` for a simple and one following the original dictionary from [StableBaselines](https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/gail/dataset/record_expert.html#generate_expert_traj).

## v0.1.0

### FIx:
* Sometimes, when the policy did not reach the goal and the enjoy function returned `False`, the `Controller` would not execute the enjoy function again

## First release
First release for IL-Datasets.
The missing features (such as documentation) list is in the README.md. If you have any issues with this release be sure to open an issue or contact me 😄 
