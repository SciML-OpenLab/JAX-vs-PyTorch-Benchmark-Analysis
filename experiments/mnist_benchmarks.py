from models.pytorch_cnn import train as torch_train
from models.jax_cnn import train as jax_train
from .cnn_benchmark import multi_run

def run_mnist(config):
    return {
        "PyTorch": multi_run(torch_train, config),
        "JAX": multi_run(jax_train, config)
    }
