import jax
import jax.numpy as jnp
from jax import grad, jit
import time

def init_params(key, d_model=32, num_heads=2, num_classes=10):
    keys = jax.random.split(key, 6)

    # Attention weights
    Wq = jax.random.normal(keys[0], (d_model, d_model)) * 0.01
    Wk = jax.random.normal(keys[1], (d_model, d_model)) * 0.01
    Wv = jax.random.normal(keys[2], (d_model, d_model)) * 0.01

    # Feedforward
    W1 = jax.random.normal(keys[3], (d_model, d_model)) * 0.01
    b1 = jnp.zeros((d_model,))
    W2 = jax.random.normal(keys[4], (d_model, num_classes)) * 0.01
    b2 = jnp.zeros((num_classes,))

    return (Wq, Wk, Wv, W1, b1, W2, b2)

def attention(q, k, v):
    d_k = q.shape[-1]

    scores = jnp.matmul(q, k.T) / jnp.sqrt(d_k)
    weights = jax.nn.softmax(scores, axis=-1)

    return jnp.matmul(weights, v)

def forward(params, x):
    Wq, Wk, Wv, W1, b1, W2, b2 = params

    # Linear projections
    q = jnp.dot(x, Wq)
    k = jnp.dot(x, Wk)
    v = jnp.dot(x, Wv)

    # Attention
    attn_out = attention(q, k, v)

    # Feedforward
    h = jnp.dot(attn_out, W1) + b1
    h = jax.nn.relu(h)

    # Pool over sequence (mean)
    h = jnp.mean(h, axis=0)

    logits = jnp.dot(h, W2) + b2

    return logits

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)

    log_probs = jax.nn.log_softmax(logits)
    return -jnp.sum(one_hot * log_probs)

@jit
def update(params, x, y, lr):
    grads = grad(loss_fn)(params, x, y)
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)

def generate_data(seq_len=32, feature_dim=32):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (seq_len, feature_dim))
    y = jax.random.randint(key, (), 0, 10)
    return x, y

def train(config):
    key = jax.random.PRNGKey(42)
    params = init_params(key)

    start = time.time()

    for epoch in range(config["epochs"]):
        for _ in range(20):  # simulate batches
            x, y = generate_data()
            params = update(params, x, y, config["lr"])

    end = time.time()

    final_loss = float(loss_fn(params, *generate_data()))

    return {
        "time": end - start,
        "final_loss": final_loss
    }
