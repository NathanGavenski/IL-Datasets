# Creating a new dataset

Creation of new Datasets use the `Controller` class, which requires two different function to work:
* an `enjoy` function, which uses the agent to interact with the environment and record an episode); and
* a `collate` function, which puts all files created by the `enjoy` function into a dataset file.

## Default functions

The `imitation_datasets` module provides a set of default functions, so you don't need to implement an `enjoy` and a `collate` function in every project.
The resulting dataset will be a `NpzFile` with the following data:

```{python}
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
```{python}
# command: $ python <script> --game cartpole --threads 4 --episodes 1000 --mode all
from imitation_datasets.functions import baseline_enjoy, baseline_collate
from imitation_datasets.controller import Controller
from imitation_datasets.args import get_args

args = get_args()
controller = Controller(baseline_enjoy, baseline_collate, args.episodes, args.threads)
controller.start(args)
```

## Converting to a HuggingFace dataset

If you want to upload the dataset to HuggingFace, the `imitation_datasets` module also provides a function for converting the `NpzFile` into `jsonl` format.

```{python}
from imitation_datasets.dataset.huggingface import baseline_to_huggingface

baseline_to_huggingface("./path/to/NpZFile", "./new/path")
```
