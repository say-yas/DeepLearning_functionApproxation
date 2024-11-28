import tensorflow as tf
# from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# class ReLUNetwork(tf.keras.Model):
#     def __init__(self, num_neurons):
#         super(ReLUNetwork, self).__init__()
#         self.dense1 = tf.keras.layers.Dense(num_neurons, activation='relu')
#         self.dense2 = tf.keras.layers.Dense(1)

#     def call(self, inputs):
#         x = self.dense1(inputs)
#         return self.dense2(x)


class ReLUNetwork(tf.keras.Model):
    """Defines a standard fully-connected network in TensorFlow"""
    
    def __init__(self, num_input, num_output, num_hidden, num_layers):
        super().__init__()


        self.fcs = tf.keras.Sequential([
            tf.keras.layers.Dense(num_hidden, activation="relu", input_shape=(num_input,))
        ])
        
        self.fch = tf.keras.Sequential([
            tf.keras.layers.Dense(num_hidden, activation="relu")
            for _ in range(num_layers - 1)
        ])
        
        self.fce = tf.keras.layers.Dense(num_output)

    def call(self, inputs):
        x = self.fcs(inputs)
        x = self.fch(x)
        return self.fce(x)

def convex_function(x):
    return x**2  # Example convex function

def train_network(model, x_train, y_train, errs=[], epochs=1000):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = tf.reduce_mean(tf.square(predictions - y))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    for epoch in range(epochs):
        loss = train_step(x_train, y_train)
        errs.append(loss.numpy())

        if loss.numpy() < 1e-5:
            print("Loss less than 1e-5, stopping training.")
            break

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")


try: 
    # Generate training data
    x_train = np.linspace(-1, 1, 100).reshape(-1, 1).astype(np.float32)
    y_train = convex_function(x_train)

    # Create and train the model
    model = ReLUNetwork(1, 1, 16, 2)
    errors = []
    train_network(model, x_train, y_train, errors, epochs=2000)


    plt.figure(figsize=(7,3.5))
    plt.yscale("log")  
    plt.grid()
    plt.plot(np.array(errors), 'b-', label="Loss")
    plt.show()

    # Generate test data for plotting
    x_test = np.linspace(-1.1, 1.1, 200).reshape(-1, 1).astype(np.float32)
    y_test = convex_function(x_test)
    y_pred = model(x_test).numpy()

    # Plotting
    plt.figure(figsize=(7, 4))
    plt.xlim(-1.1, 1.1)
    plt.plot(x_test, y_test, label='True Function')
    plt.plot(x_test, y_pred, label='ReLU Network Approximation')
    plt.scatter(x_train, y_train, color='red', s=10, label='Training Data')
    plt.legend()
    plt.title('ReLU Network Approximation of a Convex Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()
except:
    raise ValueError("Error in training the RELU network tensorflow model")