import numba
import numpy as np
from sklearn.utils import check_random_state
from umap.umap_ import INT32_MAX, INT32_MIN
from umap.utils import tau_rand


def setup_rng(random_state=42):
    """return RNG state needed for Tausworthe PRNG"""
    random_state = check_random_state(random_state)
    return random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)


def setup_rngn(n, random_state=42):
    random_state = check_random_state(random_state)
    return random_state.randint(INT32_MIN, INT32_MAX, (n, 3)).astype(np.int64)


@numba.jit(nopython=True, fastmath=True)
def tau_rand_norm(rng_state, mean=0.0, sdev=1.0):
    """Return normally distributed random number"""
    u1 = tau_rand(rng_state)
    u2 = tau_rand(rng_state)
    return (bmr(u1, u2) * sdev) + mean


@numba.jit(nopython=True, fastmath=True)
def bmr(u1, u2):
    """Rectangular Box-Muller Transform converting two uniformly distributed random
    numbers to one normally distributed"""
    r = np.sqrt(-2 * np.log(u1))
    theta = 2 * np.pi * u2
    return r * np.cos(theta)
