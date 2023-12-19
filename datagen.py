import numpy as np
from numpy.random import default_rng

def add_noise(y, noise=0.1, seed=None):
    rng = default_rng(seed)
    
    return y * rng.choice([-1, 1], p=[noise, 1 - noise], size=len(y))


def linearly_separable_dataset(N, seed=None):
    rng = default_rng(seed)
    intercepts = rng.random(2)
    a, b = -intercepts[1]/intercepts[0], intercepts[1]

    X = rng.random((N, 2))
    y = np.where(X[:, 1] >= a*X[:, 0] + b, 1, -1)

    return X, y


def sinusoidal_dataset(N, amplitude=0.2, frequency=10, offset=0.5, seed=None):
    rng = default_rng(seed)
    X = rng.random((N, 2))

    y = np.where(X[:, 1] >= amplitude*np.sin(frequency*X[:, 0]) + offset, 1, -1)

    return X, y


def radially_separable_dataset(N, radius=0.5, seed=None):
    rng = default_rng(seed)
    X = rng.uniform(low=-1.0, high=1.0, size=(N, 2))

    y = np.where(X[:, 0]**2 + X[:, 1]**2 <= radius**2, 1, -1)

    return X, y


def sample_mental_image(mental_image, n, seed=None):
    rng = default_rng(seed)
    
    return rng.binomial(1, mental_image, (n, len(mental_image)))