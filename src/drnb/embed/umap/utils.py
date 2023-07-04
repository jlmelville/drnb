import numba
import numpy as np


@numba.njit(
    "f4(f4[::1],f4[::1])",
    fastmath=True,
    cache=True,
    locals={
        "result": numba.types.float32,
        "diff": numba.types.float32,
        "dim": numba.types.intp,
    },
)
def rdist(x, y):
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff

    return result


@numba.njit(fastmath=True, cache=True)
def euclidean(x, y):
    return np.sqrt(rdist(x, y))


@numba.njit()
def clip(val):
    if val > 4.0:
        return 4.0
    if val < -4.0:
        return -4.0
    return val
