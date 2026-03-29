from models.pytorch_transformer import train as torch_train
from models.jax_transformer import train as jax_train
from .cnn_benchmark import multi_run

def run_transformer(config):
    return {
        "PyTorch": multi_run(torch_train, config),
        "JAX": multi_run(jax_train, config)
    }
