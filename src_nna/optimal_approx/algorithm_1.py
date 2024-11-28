from scipy.optimize import minimize, fsolve
import numpy as np
import sympy as sp


def exp(x):
    return np.exp(x)


def x_sq(x):
    return np.power(x, 2)


def x_cube(x):
    return np.power(x, 3)


def deriv(x, f):
    if f == exp:
        return f(x)
    elif f == x_sq:
        return 2 * x
    elif f == x_cube:
        return 3 * np.power(x, 2)


def second_deriv(x, f, sign=1.0):
    if f == exp:
        return sign * f(x)
    elif f == x_sq:
        return sign * 2
    elif f == x_cube:
        return sign * 6 * x


def optim_c(a_vec, b_vec, f):
    func = lambda x: deriv(x, f) - ((f(b_vec) - f(a_vec)) / (b_vec - a_vec))
    solution = fsolve(func, [(a_vec[0] + b_vec[-1]) / 2] * len(a_vec))
    return solution


def optim_d(a_vec, b_vec, c_vec, f):
    func = lambda x: deriv(x, f) - ((f(c_vec) - f(a_vec)) / (c_vec - a_vec))
    solution = fsolve(func, [(a_vec[0] + b_vec[-1]) / 2] * len(a_vec))
    return solution


def optimal_approximation_errors(a_vec, b_vec, f):
    c_vec = optim_c(a_vec, b_vec, f)
    d_vec = optim_d(a_vec, b_vec, c_vec, f)
    f_prime_c = deriv(c_vec, f)
    f_prime_d = deriv(d_vec, f)
    delta_s_optimal_vec = ((c_vec - a_vec) / 2) * (f_prime_c - f_prime_d)
    return delta_s_optimal_vec, c_vec, d_vec


def optimal_segment_param(a_vec, b_vec, c_vec, f):
    A = (f(b_vec) - f(a_vec)) / (b_vec - a_vec)
    B = ((f(a_vec) + f(c_vec)) / 2) - (((a_vec + c_vec) / 2) * A)
    return A, B


def max_optimal_approximation_error(delta_s_optimal_vec):
    argmax = np.argmax(delta_s_optimal_vec)
    return delta_s_optimal_vec[argmax]


def min_optimal_approximation_error(delta_s_optimal_vec):
    argmin = np.argmin(delta_s_optimal_vec)
    return delta_s_optimal_vec[argmin]


def calculate_optimal_segments(A, B, domains):
    x = sp.Symbol("x")
    segments = []
    for i in range(len(A)):
        segments.append((A[i] * x + B[i], ((x >= domains[i][0]) & (x < domains[i][1]))))
    return sp.Piecewise(*segments)


def find_segment_params(current_a_vec, current_b_vec, z_vec):
    domains = [(current_a_vec[i], current_b_vec[i]) for i in range(len(current_a_vec))]
    for i in range(len(domains)):
        if i == 0:
            domains[i] = (domains[i][0], domains[i][1] + z_vec[i])
        elif i == len(domains) - 1:
            domains[i] = (domains[i][0] + z_vec[i - 1], domains[i][1])
        else:
            domains[i] = (domains[i][0] + z_vec[i - 1], domains[i][1] + z_vec[i])
    a_new_vec = np.array([domain[0] for domain in domains])
    b_new_vec = np.array([domain[1] for domain in domains])
    return a_new_vec, b_new_vec, domains


class Callback:
    def __init__(self, a_vec, b_vec, fn):
        self._prev_min_err = -np.inf
        self._prev_max_err = np.inf
        self._current_max_val = None
        self._current_min_val = None
        self.a_vec = a_vec
        self.b_vec = b_vec
        self.fn = fn

    def __call__(self, z_vec):
        a_new_vec, b_new_vec, domains = find_segment_params(
            self.a_vec, self.b_vec, z_vec
        )
        delta_s_optimal_vec, c_vec, d_vec = optimal_approximation_errors(
            a_new_vec, b_new_vec, self.fn
        )
        self._current_max_val = max_optimal_approximation_error(delta_s_optimal_vec)
        self._current_min_val = min_optimal_approximation_error(delta_s_optimal_vec)
        if abs(self._current_max_val - self._current_min_val) >= abs(
            self._prev_max_err - self._prev_min_err
        ):
            return True

        self._prev_max_err = self._current_max_val
        self._prev_min_err = self._current_min_val
        return False


def find_optimal_boundaries(z_vec, a_vec, b_vec, f):
    a_new_vec, b_new_vec, domains = find_segment_params(a_vec, b_vec, z_vec)

    delta_s_optimal_vec, c_vec, d_vec = optimal_approximation_errors(
        a_new_vec, b_new_vec, f
    )
    max_optimal_error = max_optimal_approximation_error(delta_s_optimal_vec)
    min_optimal_error = min_optimal_approximation_error(delta_s_optimal_vec)
    return max_optimal_error - min_optimal_error


def optimal_approximation_algorithm(target_fn, a, b, n_segments):
    segments_domain = [
        (
            np.round(a + (((b - a) / n_segments) * (i - 1)), 4),
            np.round(a + (((b - a) / n_segments) * (i)), 4),
        )
        for i in range(1, n_segments + 1)
    ]

    a_vec = np.array([domain[0] for domain in segments_domain])
    b_vec = np.array([domain[1] for domain in segments_domain])
    initial_z_vec = np.array(0 * np.ones(n_segments - 1))

    z_vec_optimal = minimize(
        find_optimal_boundaries,
        initial_z_vec,
        args=(a_vec, b_vec, target_fn),
        callback=Callback(a_vec, b_vec, target_fn),
    )

    a_new_vec, b_new_vec, new_domains = find_segment_params(
        a_vec, b_vec, z_vec_optimal.x
    )
    delta_s_optimal_vec, c_vec, d_vec = optimal_approximation_errors(
        a_new_vec, b_new_vec, target_fn
    )

    A, B = optimal_segment_param(a_new_vec, b_new_vec, c_vec, target_fn)

    f_approx = calculate_optimal_segments(A, B, new_domains)

    return f_approx, delta_s_optimal_vec


def calculate_bounds(f, a, b, n):
    min_second_deriv = minimize(second_deriv, [0], args=(f, 1.0), bounds=((a, b),))
    max_second_deriv = minimize(second_deriv, [0], args=(f, -1.0), bounds=((a, b),))
    lower_bound = (((b - a) ** 2 / 16) * second_deriv(min_second_deriv.x[0], f)) / (
        n * n
    )
    upper_bound = (((b - a) ** 2 / 16) * second_deriv(max_second_deriv.x[0], f)) / (
        n * n
    )
    return np.round(lower_bound, 4), np.round(upper_bound, 4)
