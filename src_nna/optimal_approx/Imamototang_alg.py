import numpy as np

class PW_approx:
    def __init__(self, m, c, t, alpha, epsilon):
        self.m = m
        self.c = c
        self.t = t
        self.alpha = alpha
        self.epsilon = epsilon

    def __call__(self, x):
        return max([m_i*x+c_i for m_i, c_i in zip(self.m, self.c)])
    
def immamoto_optimal_approximation(f_x, f_prime_x, alpha_0, alpha_N, N, gamma=1e-4):
    j = 0
    delta = 1
    t = dict()
    alpha = dict()
    epsilon = dict()
    d = dict()
    t[j] = [alpha_0 + (((i / N) + 0.5)*(alpha_N - alpha_0)) for i in range(N)]

    while True:
        alp = [
            (
                (
                    f_x(t[j][i - 1])
                    - f_x(t[j][i])
                    + f_prime_x(t[j][i]) * t[j][i]
                    - f_prime_x(t[j][i - 1]) * t[j][i - 1]
                )
                / (f_prime_x(t[j][i]) - f_prime_x(t[j][i - 1]))
            )
            for i in range(1, N)
        ]
        alpha[j] = [alpha_0] + alp + [alpha_N]
        epsilon[j] = [
            f_prime_x(t[j][i]) * (alpha[j][i] - t[j][i])
            + f_x(t[j][i])
            - f_x(alpha[j][i])
            for i in range(N)
        ]
        epsilon[j] = epsilon[j] + [
            f_prime_x(t[j][N - 1]) * (alpha[j][N] - t[j][N - 1])
            + f_x(t[j][N - 1])
            - f_x(alpha[j][N])
        ]
        abs_epsilon_j = np.abs(epsilon[j])
        if (max(abs_epsilon_j) / (min(abs_epsilon_j) + 1e-12)) - 1 < gamma:
            break
        if (j - 1 in epsilon) and max(abs_epsilon_j) > max(np.abs(epsilon[j - 1])):
            delta = delta / 2
            j = j - 1
        
        d[j] = [
            delta
            * (
                (epsilon[j][i + 1] - epsilon[j][i])
                / (
                    (epsilon[j][i + 1] / (alpha[j][i + 1] - t[j][i] + 1e-12))
                    + (epsilon[j][i] / (t[j][i] - alpha[j][i] + 1e-12))
                )
            )
            for i in range(N)
        ]
        t[j + 1] = [t[j][i] + d[j][i] for i in range(N)]
        j = j + 1
    epsilon_final = 0.5 * min(epsilon[j])
    m = [ f_prime_x(t[j][i]) for i in range(N) ]
    b = [ -1*f_prime_x(t[j][i])*t[j][i] + f_x(t[j][i]) - epsilon_final for i in range(N) ]
    g = PW_approx(m, b, t[j], alpha[j], epsilon_final)
    return g, np.abs(epsilon[j])
