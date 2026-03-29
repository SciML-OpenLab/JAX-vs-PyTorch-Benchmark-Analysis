import numpy as np
from models.pytorch_model import train as torch_train
from models.jax_model import train as jax_train

NUM_RUNS = 5

def run_experiment(name, train_fn, config):
    result = train_fn(config)
    return result

def multi_run(train_fn, config):
    times = []
    losses = []

    for i in range(NUM_RUNS):
        print(f"Run {i+1}/{NUM_RUNS}")
        result = run_experiment(f"run_{i}", train_fn, config)
        times.append(result["time"])
        losses.append(result["final_loss"])

    return {
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "mean_loss": float(np.mean(losses)),
        "all_times": times
    }

def run_full_benchmark(config):
    print("\n Running PyTorch Benchmark")
    torch_stats = multi_run(torch_train, config)

    print("\n Running JAX Benchmark")
    jax_stats = multi_run(jax_train, config)

    return {
        "PyTorch": torch_stats,
        "JAX": jax_stats
    }
