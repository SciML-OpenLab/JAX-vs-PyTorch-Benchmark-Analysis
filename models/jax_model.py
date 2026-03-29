import jax
import jax.numpy as jnp
import time
from jax import grad, jit

def init_params():
    key = jax.random.PRNGKey(0)
    W1 = jax.random.normal(key, (100, 64))
    b1 = jnp.zeros((64,))
    W2 = jax.random.normal(key, (64, 1))
    b2 = jnp.zeros((1,))
    return (W1, b1, W2, b2)

def forward(params, x):
    W1, b1, W2, b2 = params
    h = jnp.dot(x, W1) + b1
    h = jax.nn.relu(h)
    return jnp.dot(h, W2) + b2

def loss_fn(params, x, y):
    preds = forward(params, x)
    return jnp.mean((preds - y) ** 2)

@jit
def update(params, x, y, lr):
    grads = grad(loss_fn)(params, x, y)
    return jax.tree_map(lambda p, g: p - lr * g, params, grads)

def train(config):
    params = init_params()

    X = jnp.array(jax.random.normal(jax.random.PRNGKey(1), (1000, 100)))
    y = jnp.array(jax.random.normal(jax.random.PRNGKey(2), (1000, 1)))

    start = time.time()

    for _ in range(config["epochs"]):
        params = update(params, X, y, config["lr"])

    end = time.time()

    return {
        "time": end - start,
        "final_loss": float(loss_fn(params, X, y))
    }
