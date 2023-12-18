# Imitation Learning Datasets

Hi, welcome to the Imitation Learning (IL) Datasets. 
Something that always bothered me was how difficult it is to find good weights for an expert, trying to create a dataset for different state-of-the-art methods, and having to run all methods due to no common datasets. 
For these reasons, I've created IL-Datasets, in an effort to make it more accessible for researchers to create datasets using experts from the Hugging Face.
IL-Datasets provides teacher weights for different environments, a multi-threading solution for creating datasets faster, datasets for a set of environments, and a benchmark for common imitation learning methods.

**This project is under development. If you are interested in helping, feel free to contact [me](https://nathangavenski.github.io/).**

## Main Features

* Dataset creation with StableBaselines from HuggingFace weights.
* Dataset creation with user custom policies.
* Readily available datasets for common benchmark environments.
* Benchmark results for all implemented methods in the published datasets.
