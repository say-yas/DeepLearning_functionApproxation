import numpy as np
from src_nna.axon_approx.src.axon_approximation import axon_algorithm
from src_nna.axon_approx.src.axon_model_jax import train_random_model_jax
from tqdm import tqdm

class prettyfloat(float):
    def __repr__(self):
        return "%0.4f" % self
    

xs = np.linspace(0, 1, 1000)[:,None]

# classical training with random initialization
errs_rnd = []
idx_print = 10
for k in tqdm(range(1,20)):
    err_k = train_random_model_jax(xs, lambda x: x**2, K=k, num_epochs=550, num_iters=170, learning_rate=6e-2)
    err_k = [err for err in err_k if not np.isnan(err)] # to avoid nans when min is searched
    errs_rnd.append(min(err_k))
    if k % idx_print == 0:
        print(f"Epoch {k}, Loss: {errs_rnd[k-3:k]}")

# print(errs_rnd)
file_save = './Data/axontest_jax_x2.npy'
np.save(file_save, errs_rnd)

import matplotlib.pyplot as plt
plt.plot(errs_rnd, label='Jax: f(x)= x^2')
filename = 'Figs/axontest_jax_x2.pdf'
plt.savefig(filename)
plt.show()