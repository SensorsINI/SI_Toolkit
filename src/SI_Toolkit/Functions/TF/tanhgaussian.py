import matplotlib
matplotlib.use('MacOSX')

import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import tensorflow as tf

def tanh_gaussian(x, sigma):
    return np.tanh(x) * np.exp(-x ** 2 / (2 * sigma ** 2))


def tanh_gaussian_tf(x, sigma=1.0):
    return tf.math.tanh(x) * tf.exp(-(x**2) / (2 * sigma**2))


def find_max_tanh_gaussian(sigma, search_interval=(0, None)):
    """
    Numerically calculate the maximum of the function tanh(x) * Gaussian(x, sigma).

    Since tanh is an odd function and the Gaussian is even, the product is odd;
    thus, the maximum occurs for positive x. This routine finds the x value and the maximum.

    Parameters:
    sigma : float
        The standard deviation for the Gaussian function.
    search_interval : tuple (optional)
        A tuple (x_min, x_max) specifying the interval to search for the maximum.
        If x_max is None, it defaults to 10 * sigma.

    Returns:
    x_max : float
        The x-value where the function attains its maximum.
    f_max : float
        The maximum function value.
    """
    # Set default upper bound if None is provided
    x_min = search_interval[0]
    x_max = search_interval[1] if search_interval[1] is not None else 10 * sigma

    # We seek the maximum by minimizing the negative of the function
    negative_func = lambda x: -tanh_gaussian(x, sigma)

    # Use scipy's bounded minimization within [x_min, x_max]
    result = minimize_scalar(negative_func, bounds=(x_min, x_max), method='bounded')

    if result.success:
        x_opt = result.x
        f_opt = tanh_gaussian(x_opt, sigma)
        return x_opt, f_opt
    else:
        raise RuntimeError("Optimization did not converge.")


def normed_tanh_gaussian_factory(sigma):
    _, max_value = find_max_tanh_gaussian(sigma)
    normed_tanh_gaussian = lambda x: tanh_gaussian_tf(max_value * x, sigma) / max_value
    return normed_tanh_gaussian

# Example usage:
if __name__ == "__main__":
    sigma = 5.0  # You can change sigma as needed
    x_max, f_max = find_max_tanh_gaussian(sigma)
    print("Maximum value of tanh*Gaussian:")
    print("x_max =", x_max)
    print("f_max =", f_max)

    range = 50
    x_vals = np.linspace(-range, range, 500)
    x_tensor = tf.convert_to_tensor(x_vals, dtype=tf.float32)
    y_tanh = tf.math.tanh(x_tensor)
    normed_tanh_gaussian = normed_tanh_gaussian_factory(sigma)
    y_gaussian_tanh = normed_tanh_gaussian(x_tensor)
    plt.plot(x_vals, y_tanh.numpy(), label='tanh', linestyle='--')
    plt.plot(x_vals, y_gaussian_tanh.numpy(), label='gaussian_tanh (σ=1.0)')
    plt.title("Standard tanh vs. Gaussian-attenuated tanh")
    plt.xlabel("x")
    plt.ylabel("Activation")
    plt.grid(True)
    plt.legend()
    plt.show()

