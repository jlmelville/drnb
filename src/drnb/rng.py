import numpy as np
from sklearn.utils import check_random_state
from umap.umap_ import INT32_MAX, INT32_MIN


def setup_rng(random_state=42):
    """return RNG state needed for Tausworthe PRNG"""
    random_state = check_random_state(random_state)
    return random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)


def setup_rngn(n, random_state=42):
    random_state = check_random_state(random_state)
    return random_state.randint(INT32_MIN, INT32_MAX, (n, 3)).astype(np.int64)
