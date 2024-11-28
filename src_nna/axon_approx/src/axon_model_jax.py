import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import optax
import numpy as np
from functools import partial
from tqdm import tqdm

def init_network(x, y, num_basis_fun, key):
    x_shape = x.shape[1]
    
    bs = jnp.concatenate([jnp.ones((x.shape[0], 1)), x], axis=1)
    bs, r = jnp.linalg.qr(bs)
    r_inv = jnp.linalg.inv(r)
    
    layers = []
    for i in range(num_basis_fun - x_shape - 1):
        key, subkey = random.split(key)
        layers.append({
            'weight': random.normal(subkey, (i + x_shape + 1, 1))
        })
    
    bas = build_basis_jax(layers, bs)
    c = jnp.linalg.lstsq(bas['basis'], y)[0]
    
    return {
        'layers': layers,
        'r_inv': r_inv,
        'c': c,
        'norms': bas['norms'],
        'coefs': bas['coefs']
    }

def build_basis_jax(layers, x):
    norms = []
    coefs = []
    for l in layers:
        new_x = jax.nn.relu(jnp.dot(x, l['weight']))
        coefs.append(jnp.dot(x.T, new_x).reshape(-1))
        new_x = new_x - jnp.dot(x, jnp.dot(x.T, new_x))
        norms.append(jnp.linalg.norm(new_x))
        new_x = new_x / norms[-1]
        x = jnp.concatenate([x, new_x], axis=1)
    return {'basis': x, 'norms': norms, 'coefs': coefs}

@jit
def get_basis_jax(params, x):
    out = jnp.concatenate([jnp.ones((x.shape[0], 1)), x], axis=1)
    x = jnp.dot(out, params['r_inv'])
    
    for i, l in enumerate(params['layers']):
        new_x = jax.nn.relu(jnp.dot(x, l['weight']))
        for coef, norm in zip(params['coefs'][i:i+1], params['norms'][i:i+1]):
            new_x = new_x - jnp.dot(x, coef)[:, None]
            new_x = new_x / norm
        x = jnp.concatenate([x, new_x], axis=1)
    return x

@jit
def predict(params, x):
    x = get_basis_jax(params, x)
    return jnp.dot(x, params['c'])

@jit
def loss_fn(params, x, y):
    pred = predict(params, x)
    return jnp.mean((pred - y) ** 2)

# @jit
# def update(params, opt_state, x, y):
#     loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
#     updates, opt_state = optimizer.update(grads, opt_state)
#     params = optax.apply_updates(params, updates)
#     return params, opt_state, loss

def train_random_model_jax(xs, f, K, num_epochs, num_iters, learning_rate=1e-1):
    fs = f(xs).flatten()

    xs = jnp.array(xs.astype(jnp.float32))
    fs = jnp.array(fs.astype(jnp.float32))

    # Define the optimizer
    optimizer = optax.adam(learning_rate=learning_rate)

    # Define the update function with the optimizer
    @jit
    def update(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    errors = []
    for j in tqdm(range(num_iters)):  # train several times
        key = random.PRNGKey(j)
        params = init_network(xs, fs, num_basis_fun=K+xs.shape[-1]+1, key=key)
        
        # Initialize the optimizer state
        opt_state = optimizer.init(params)

        for i in range(num_epochs):
            params, opt_state, loss = update(params, opt_state, xs, fs)

        pred = predict(params, xs)
        error = jnp.linalg.norm(pred - fs) / jnp.linalg.norm(fs)
        # if np.isnan(error):
        #     print("Error is nan")
        #     print(np.isnan(pred))
        errors.append(error.item())
    return errors

# # Usage example:
# xs = jnp.linspace(-1, 1, 100).reshape(-1, 1)
# f = lambda x: x**2
# errors = train_random_model_jax(xs, f, K=5, num_epochs=1000)

# import matplotlib.pyplot as plt
# print(errors)
# plt.plot(errors)
# plt.show()