from dataclasses import dataclass

import numba
import numpy as np

from drnb.distances import distance_function
from drnb.eval import EvalResult

from .base import EmbeddingEval


def stress(X, coords, metric="euclidean"):
    xdfun = distance_function(metric)
    ydfun = distance_function("euclidean")

    return _stress(X, coords, xdfun, ydfun)


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _stress(X, Y, xdfun, ydfun):
    nobs, _ = X.shape
    stressv = np.zeros(nobs)
    # pylint: disable=not-an-iterable
    for i in numba.prange(nobs):
        for j in range(i + 1, nobs):
            rij = xdfun(X[i], X[j])
            dij = ydfun(Y[i], Y[j])
            stressv[i] += (rij - dij) ** 2
    return np.sqrt(np.mean(stressv))


@dataclass
class StressEval(EmbeddingEval):
    metric: str = "euclidean"

    def evaluate(self, X, coords, ctx=None):
        result = stress(X, coords, self.metric)

        return EvalResult(
            eval_type="Stress",
            label=str(self),
            info=dict(metric=self.metric),
            value=result,
        )

    def __str__(self):
        return f"stress-{self.metric}"
