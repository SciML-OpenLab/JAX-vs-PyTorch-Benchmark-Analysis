import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np
import time
from torchvision import datasets, transforms

def load_mnist():
    transform = transforms.ToTensor()
    dataset = datasets.MNIST('.', train=True, download=True, transform=transform)

    X = []
    y = []

    for img, label in dataset:
        X.append(np.array(img).reshape(-1))  # flatten 28x28 → 784
        y.append(label)

    X = jnp.array(np.stack(X))
    y = jnp.array(np.array(y))

    return X, y

def init_params(key):
    k1, k2 = jax.random.split(key)

    W1 = jax.random.normal(k1, (784, 128)) * 0.01
    b1 = jnp.zeros((128,))
    W2 = jax.random.normal(k2, (128, 10)) * 0.01
    b2 = jnp.zeros((10,))

    return (W1, b1, W2, b2)

def forward(params, x):
    W1, b1, W2, b2 = params

    h = jnp.dot(x, W1) + b1
    h = jax.nn.relu(h)
    logits = jnp.dot(h, W2) + b2

    return logits

def cross_entropy_loss(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)

    log_probs = jax.nn.log_softmax(logits)
    loss = -jnp.sum(one_hot * log_probs, axis=1)

    return jnp.mean(loss)

@jit
def update(params, x, y, lr):
    grads = grad(cross_entropy_loss)(params, x, y)
    return jax.tree_map(lambda p, g: p - lr * g, params, grads)

def train(config):
    X, y = load_mnist()

    key = jax.random.PRNGKey(0)
    params = init_params(key)

    batch_size = 64
    num_batches = X.shape[0] // batch_size

    start = time.time()

    for epoch in range(config["epochs"]):
        for i in range(num_batches):
            batch_x = X[i*batch_size:(i+1)*batch_size]
            batch_y = y[i*batch_size:(i+1)*batch_size]

            params = update(params, batch_x, batch_y, config["lr"])

    end = time.time()

    final_loss = float(cross_entropy_loss(params, X[:1000], y[:1000]))

    return {
        "time": end - start,
        "final_loss": final_loss
    }
