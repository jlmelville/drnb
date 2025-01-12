from dataclasses import dataclass

import numba
import numpy as np

from drnb.distances import DistanceFunc, distance_function
from drnb.embed.context import EmbedContext
from drnb.eval.base import EmbeddingEval, EvalResult


def stress(
    X: np.ndarray,
    coords: np.ndarray,
    metric: str = "euclidean",
) -> float:
    """Compute the stress of an embedding: the sum of squared differences between
    pairwise distances in the original space and the embedding space."""
    xdfun = distance_function(metric)
    ydfun = distance_function("euclidean")

    return _stress(X, coords, xdfun, ydfun)


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _stress(
    X: np.ndarray,
    Y: np.ndarray,
    xdfun: DistanceFunc,
    ydfun: DistanceFunc,
) -> float:
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
    """Compute the stress of an embedding: the sum of squared differences between
    pairwise distances in the original space and the embedding space.

    Attributes:
        metric: Distance metric to use for calculating the stress.
    """

    metric: str = "euclidean"

    def evaluate(
        self, X: np.ndarray, coords: np.ndarray, _: EmbedContext = None
    ) -> EvalResult:
        result = stress(X, coords, self.metric)

        return EvalResult(
            eval_type="Stress",
            label=str(self),
            info={"metric": self.metric},
            value=result,
        )

    def __str__(self):
        return f"stress-{self.metric}"
