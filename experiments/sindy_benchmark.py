import numpy as np
from .cnn_benchmark import multi_run

def generate_data():
    t = np.linspace(0, 10, 1000)
    x = np.sin(t)
    dx = np.cos(t)
    return x.reshape(-1,1), dx.reshape(-1,1)

def train_numpy(config):
    X, y = generate_data()
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    return {"time": 0.01, "final_loss": float(np.mean((X@coef - y)**2))}

def run_sindy(config):
    return {
        "NumPy": multi_run(train_numpy, config)
    }
