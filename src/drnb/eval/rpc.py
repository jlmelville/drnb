from dataclasses import dataclass

import numpy as np
import scipy.stats

from .base import EmbeddingEval
from .triplets import calc_distances, get_triplets, validate_triplets


def random_pair_correl_eval(
    X, X_new, triplets=None, random_state=None, n_triplets_per_point=5
):
    n_obs = X.shape[0]
    if triplets is None:
        triplets = get_triplets(
            X, seed=random_state, n_triplets_per_point=n_triplets_per_point
        )
    else:
        validate_triplets(triplets, n_obs)
        n_triplets_per_point = triplets.shape[1]

    anchors = np.arange(n_obs).reshape((-1, 1, 1))
    bpairs = np.broadcast(anchors, triplets)
    d_X = calc_distances(X, bpairs)

    bpairs = np.broadcast(anchors, triplets)
    d_Xnew = calc_distances(X_new, bpairs)

    return scipy.stats.pearsonr(d_X.flatten(), d_Xnew.flatten()).statistic


@dataclass
class RandomPairCorrelEval(EmbeddingEval):
    random_state: int = None
    n_triplets_per_point: int = 5

    def evaluate(self, X, coords, ctx=None):
        rpc = random_pair_correl_eval(
            X,
            coords,
            random_state=self.random_state,
            n_triplets_per_point=self.n_triplets_per_point,
        )
        return ("rpc", rpc)

    def __str__(self):
        return (
            "Random Pair Correlation "
            + f"num triplets per point: {self.n_triplets_per_point}"
        )
