# Benchmarking

All benchmark results are available in the [benchmark_results.md](https://github.com/NathanGavenski/IL-Datasets/blob/main/benchmark_results.md).
If you wish to run all benchmarks locally, feel free to run them with:
```{bash}
$ python src/bencmark/benchmark.py
```

This command will run all implemented methods with all available [datasets](https://huggingface.co/collections/NathanGavenski/imitation-learning-datasets-6542982072defaf65937432d).
Be careful, it will take a long time!
If you wish to run a single method or run them in parallel, use:
```{bash}
$ python src/bencmark/benchmark.py --methods <METHOD NAME>
```

For example:
```{bash}
$ python src/bencmark/benchmark.py --methods bc,bco
```
will run Behavioural Cloning and Behavioural Clonning from Observation.
