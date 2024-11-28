import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import optax
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def init_network_params(sizes, key):
    """Initialize the weights of a fully-connected ReLU network."""
    keys = random.split(key, len(sizes))
    return [
        {'w': random.normal(k, (m, n)) * jnp.sqrt(2.0 / m),
         'b': jnp.zeros(n)}
        for k, m, n in zip(keys, sizes[:-1], sizes[1:])
    ]

def relu(x):
    return jnp.maximum(0, x)

def network_predict(params, x):
    """Forward pass through the network."""
    activations = x
    for layer in params[:-1]:
        activations = relu(jnp.dot(activations, layer['w']) + layer['b'])
    final_layer = params[-1]
    return jnp.dot(activations, final_layer['w']) + final_layer['b']

def loss(params, x, y):
    """Mean squared error loss."""
    pred = network_predict(params, x)
    return jnp.mean((pred - y) ** 2)

@jit
def update(params, x, y, opt_state):
    """Compute gradients and update parameters."""
    grads = grad(loss)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

def convex_function(x):
    """Example convex function to approximate."""
    return x**2


try: 
    # Set up the network and optimization
    layer_sizes = [1, 16, 16, 1]  # Input dim, hidden layers, output dim
    learning_rate = 0.01
    num_epochs = 1000

    key = random.PRNGKey(0)
    params = init_network_params(layer_sizes, key)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Generate training data
    x_train = jnp.linspace(-1, 1, 100).reshape(-1, 1)
    y_train = convex_function(x_train)

    errors=[]
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        params, opt_state = update(params, x_train, y_train, opt_state)
        train_loss = loss(params, x_train, y_train)
        errors.append(train_loss)

        if train_loss < 1e-5:
            print(f"Converged at iteration {epoch}", "final error:", train_loss)
            break

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {train_loss:.4f}")

    plt.figure(figsize=(7,3.5))
    plt.yscale("log")  
    plt.grid()
    plt.plot(np.array(errors), 'b-', label="Loss")
    plt.show()

    # Generate test data and predictions
    x_test = jnp.linspace(-1.1, 1.1, 200).reshape(-1, 1)
    y_test = convex_function(x_test)
    y_pred = vmap(lambda x: network_predict(params, x))(x_test)

    # Plotting
    plt.figure(figsize=(7, 4))
    plt.xlim(-1.1, 1.1)
    plt.plot(x_test, y_test, label='True Function')
    plt.plot(x_test, y_pred, label='ReLU Network Approximation')
    plt.scatter(x_train, y_train, color='red', s=10, label='Training Data')
    plt.legend()
    plt.title('ReLU Network Approximation of a Convex Function (JAX)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()
except:
    print("Error in RELU nn based on JAX")