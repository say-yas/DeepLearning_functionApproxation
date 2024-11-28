import numpy as np
from src_nna.axon_approx.src.axon_approximation import axon_algorithm
from src_nna.axon_approx.src.axon_model_torch import train_random_model_torch

from tqdm import tqdm

xs = np.linspace(0, 1, 2000)[:,None]

# classical training with random initialization
errs_rnd = []
for k in tqdm(range(1,10)):
    err_k = train_random_model_torch(xs, lambda x: x**2, K=k, num_epochs=205, num_iters=100, learning_rate=1e-2)
    err_k = [err for err in err_k if not np.isnan(err)] # to avoid nans when min is searched
    errs_rnd.append(min(err_k))

# print(errs_rnd)
import matplotlib.pyplot as plt
plt.plot(errs_rnd, label='Torch')
plt.show()