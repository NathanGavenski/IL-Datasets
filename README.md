# IL Datasets

Hi, welcome to the Imitation Learning (IL) Datasets.
Something that always bothered me a lot was how difficult it was to find good weights for an expert, or trying to create a dataset for different state-of-the-art methods.
For this reason I've created this repository in an effort to make it more accessible for researches to create datasets using experts from the [Hugging Face](https://huggingface.co/models?pipeline_tag=reinforcement-learning).

--- 
## How does it work?

This project also works with multithreading, which should accelerate the dataset creation.
It consists of one ``Controller`` class, which requires two different functions to work: (i) a ``enjoy`` function (for the agent to play and record an episode); and a (ii) ``collate`` function (for putting all episodes together).

---

The ``enjoy`` function will receive 3 parameters and return 1:

* path: str - where the episode is going to be recorded
* experiment: Context - A class for recording all information (if you don't want to use `print` - keeping the console clear)
* expert: Policy - A model based on the [StableBaselines3](https://stable-baselines3.readthedocs.io/en/master/) `BaseAlgorithm`.

* returns: bool - Whether it was successfull or not

Obs: To use the model you can call ``predict``, the policy class already has the correct form of using it (a.k.a., how the StableBaselines3 uses).

---

The ``collate`` function will receive 2 parameters and return 1:

* path: str - where it should save the final dataset
* episodes: list\[str\] - A list of paths for each file

* returns: bool - Whether it was successfull or not


---
## Requirements

I did use Python=3.9 during development. \
All other requirements are listed in [requirements.txt](./requirements.txt).

---
## Registering new experts

If you would like to add new experts locally, you can call the [Experts](./utils/experts.py) class. It uses the following structure:

* identifier: str - A name for calling the expert.
* policy: Policy - A dataclass with:
    * name: str - Gym Environment name
    * repo_id: str - Hugging Face repo indentification
    * filename: str - Weights file name
    * threshold: float - How much reward should the episode accumulate to be considered good
    * algo: BaseAlgorithm - The class from StableBaselines3

Obs: If not using StableBaselines, the expert has to have a `predict` function that receives:

* obs: Tensor - Current environment state
* state: Tensor - Model's internal state
* deterministic: bool - If it should explore or not

---
## This repository is in development

Here is a list of the upcoming releases:

- [x] Collate function support
- [X] Support for installing as a dependency
- [ ] Module for downloading trajectories from a Hugging Face dataset 
    - [ ] Create benchmark for data
- [ ] Create actual documentation
- [X] Create some examples
- [ ] Create tests
- [X] Create GitHub workflow with pylint
 
---

## If you like this repository be sure to check my other projects:

### Development
- [An easy to use Wrapper for Tensorboard](https://github.com/NathanGavenski/Tensorboard-Wrapper)
- [A watcher for python to facilitate development of small projects](https://github.com/NathanGavenski/python-watcher)

### Academic
- [Imitating Unknown Policies via Exploration (BMVC)](https://arxiv.org/pdf/2008.05660.pdf)
- [Augmented behavioral cloning from observation (IJCNN)](https://arxiv.org/pdf/2004.13529.pdf)
- [Self-supervised imitation learning from observation (MSc dissertation)](https://repositorio.pucrs.br/dspace/bitstream/10923/17536/1/000500266-Texto%2Bcompleto-0.pdf)

