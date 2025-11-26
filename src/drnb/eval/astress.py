from dataclasses import dataclass

import numba
import numpy as np
from umap.utils import tau_rand_int

from drnb.distances import DistanceFunc, distance_function
from drnb.embed.context import EmbedContext
from drnb.eval.base import EvalResult
from drnb.rng import setup_rngn

from .base import EmbeddingEval


def approx_stress(
    X: np.ndarray,
    coords: np.ndarray,
    metric: str = "euclidean",
    n_samples: int = 3,
    random_state: int = 42,
) -> float:
    """Compute the approximate stress of an embedding: the sum of squared differences
    between pairwise distances in the original space and the embedding space, based
    on a random sample of points, n_samples per item in the embedding."""
    xdfun = distance_function(metric)
    ydfun = distance_function("euclidean")
    rng_state = setup_rngn(X.shape[0], random_state)

    return _approx_stress(X, coords, xdfun, ydfun, n_samples, rng_state)


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _approx_stress(
    X: np.ndarray,
    Y: np.ndarray,
    xdfun: DistanceFunc,
    ydfun: DistanceFunc,
    n_samples: int,
    rng_state: np.ndarray,
) -> float:
    nobs, _ = X.shape
    stressv = np.zeros(nobs)
    # pylint: disable=not-an-iterable
    for i in numba.prange(nobs):
        for _ in range(n_samples):
            j = tau_rand_int(rng_state[i]) % nobs
            if i == j:
                continue
            rij = xdfun(X[i], X[j])
            dij = ydfun(Y[i], Y[j])
            stressv[i] += (rij - dij) ** 2
    return np.sqrt(np.mean(stressv))


@dataclass
class ApproxStressEval(EmbeddingEval):
    """Compute the approximate stress of an embedding: the sum of squared differences
    between pairwise distances in the original space and the embedding space, based
    on a random sample of points, n_samples per item in the embedding.

    Attributes:
    random_state: int - random seed
    metric: str - distance metric to use (default: "euclidean")
    n_samples: int - number of samples per item in the embedding (default: 3)
    """

    random_state: int | None = None
    metric: str = "euclidean"
    n_samples: int = 3

    def evaluate(
        self, X: np.ndarray, coords: np.ndarray, _: EmbedContext | None = None
    ) -> EvalResult:
        result = approx_stress(
            X, coords, self.metric, self.n_samples, self.random_state
        )

        return EvalResult(
            eval_type="ApproxStress",
            label=str(self),
            info={"metric": self.metric, "n_samples": self.n_samples},
            value=result,
        )

    def __str__(self):
        return f"astress-{self.metric}-{self.n_samples}"
