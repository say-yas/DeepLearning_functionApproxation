import tensorflow as tf
import numpy as np
from tqdm import tqdm

class AxonNetwork_tf(tf.keras.Model):
    '''TensorFlow model for Axon architecture.'''

    def __init__(self, x, y, basis_coef_init=None, r=None, orth_coefs=None, norms=None, bas_np=None, num_basis_fun=3):
        
        super(AxonNetwork_tf, self).__init__()

        self.num_basis_fun = num_basis_fun
        self.r = None
        self.coefs = []
        self.norms = []
        self.c = None

        self.custom_layers = []  # Use a different name for layers
        if (basis_coef_init is None) or (r is None) or (orth_coefs is None) or (norms is None) or (bas_np is None):
            bs = tf.concat([tf.ones((x.shape[0], 1)), x], axis=1)
            bs, r = tf.linalg.qr(bs, full_matrices=False)
            self.r = tf.linalg.inv(r)

            for i in range(num_basis_fun - x.shape[1] - 1):
                # initializer = tf.keras.initializers.GlorotUniform(seed=None)
                self.custom_layers.append(tf.keras.layers.Dense(1, use_bias=False))#, kernel_initializer=initializer))
            
            self.build_layers = self.custom_layers
            bas = self.build_basis_tf(bs)
            
            self.c = tf.transpose(bas) @ tf.reshape(y, (-1, 1))

            
        else:
            bas = tf.convert_to_tensor(bas_np.astype(np.float32))
            self.r = tf.linalg.inv(tf.convert_to_tensor(r.astype(np.float32)))
            self.norms = norms
            self.coefs = [[tf.convert_to_tensor(coef.astype(np.float32)) for coef in c] for c in orth_coefs]

            for i in range(len(basis_coef_init)):
                self.custom_layers.append(tf.keras.layers.Dense(1, use_bias=False))
                # self.custom_layers[-1].build((None, i + x.shape[1] + 1))
                self.custom_layers[-1].set_weights([tf.reshape(basis_coef_init[i].astype(np.float32), (1, -1))])
                
            self.build_layers = self.custom_layers
            self.c = tf.transpose(bas) @ tf.reshape(y, (-1, 1))
            self.r = self.r

    def call(self, x):
        x = self.get_basis_tf(x)
        return x@self.c

    def build_basis_tf(self, x):

        for layer in self.custom_layers:

            new_x = tf.nn.relu(layer(x))
            self.coefs.append([tf.reshape((tf.transpose(x) @ new_x),-1)])
            new_x = new_x - (x@np.transpose(x)@new_x)
            norm = tf.norm(new_x)
            self.norms.append([norm])
            new_x = new_x/tf.norm(new_x)
            x = tf.concat([x, new_x], axis=1)
        return x

    def get_basis_tf(self, x):
        out = tf.concat([tf.ones((x.shape[0], 1)), x], axis=1) # v = [x, 1]
        x = out@self.r # v= R^{-1} v

        for i, layer in enumerate(self.custom_layers):
            new_x = tf.nn.relu(layer(x)) # g(w* v)
            for coef, norm in zip(self.coefs[i], self.norms[i]):
                new_x = new_x - (x@tf.reshape(coef,(-1,1))) #vk = g(w* v) -  v ak(coef)
                new_x = new_x/norm # vk = vk/ norm
            x = tf.concat([x, new_x], axis=1) # v = [v, vk]

        return x
    
def build(self, input_shape):
    initial_value = tf.keras.initializers.GlorotUniform()  # Your custom initialization logic
    self.w = self.add_weight(
        shape=(input_shape[-1], self.units),
        initializer=tf.constant_initializer(initial_value),
        trainable=True,
        name='kernel'
    )

def train_random_model_tf(xs, f, K, num_epochs, num_iters, learning_rate=1e-2):
    '''Train model with random initialization'''
    fs = f(xs).flatten()
    xs = tf.convert_to_tensor(xs, np.float32)
    fs =  tf.convert_to_tensor(fs, np.float32) 


    errors = []
    for _ in tqdm(range(num_iters)):  # train several times
        model = AxonNetwork_tf(xs, fs, num_basis_fun=K + xs.shape[-1] + 1)
        model.build(xs.shape)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss_fn = tf.keras.losses.MeanSquaredError()

        for epoch in range(num_epochs):
            with tf.GradientTape() as tape:
                pred = model(xs)  # full gradient
                loss = loss_fn(fs, pred)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        pred = model(xs)
        error = tf.norm(pred - fs) / tf.norm(fs)
        errors.append(error.numpy())
    
    return errors
