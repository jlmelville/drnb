from dataclasses import dataclass

import numpy as np
import scipy.stats

from drnb.log import log

from .base import EmbeddingEval
from .triplets import (
    cache_triplets,
    calc_distances,
    find_precomputed_triplets,
    get_triplets,
    validate_triplets,
)


def random_pair_correl_eval(
    X,
    X_new,
    triplets=None,
    random_state=None,
    n_triplets_per_point=5,
    return_triplets=False,
    X_dist=None,
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

    if X_dist is None:
        bpairs = np.broadcast(anchors, triplets)
        X_dist = calc_distances(X, bpairs)
    else:
        validate_triplets(X_dist, n_obs)

    bpairs = np.broadcast(anchors, triplets)
    Xnew_dist = calc_distances(X_new, bpairs)

    correl = scipy.stats.pearsonr(X_dist.flatten(), Xnew_dist.flatten()).statistic
    if return_triplets:
        return correl, triplets, X_dist
    return correl


@dataclass
class RandomPairCorrelEval(EmbeddingEval):
    random_state: int = None
    n_triplets_per_point: int = 5
    use_precomputed_triplets: bool = True
    metric: str = "l2"

    def evaluate(self, X, coords, ctx=None):
        idx = None
        X_dist = None
        return_triplets = False

        if self.use_precomputed_triplets and ctx is not None:
            idx, X_dist, return_triplets = find_precomputed_triplets(
                ctx, self.n_triplets_per_point, self.metric
            )
            if idx is None:
                log.info("No precomputed triplets found")
            # log a reminder if triplets could be cached but won't be for potentially
            # fixable reasons
            if return_triplets:
                if self.random_state is None:
                    log.info("Can't cache triplets because random seed is not set")
                    return_triplets = False

        rpc_result = random_pair_correl_eval(
            X,
            coords,
            random_state=self.random_state,
            triplets=idx,
            n_triplets_per_point=self.n_triplets_per_point,
            return_triplets=return_triplets,
            X_dist=X_dist,
        )
        if return_triplets:
            rpc = rpc_result[0]
            idx = rpc_result[1]
            X_dist = rpc_result[2]
            cache_triplets(
                idx,
                X_dist,
                ctx,
                self.n_triplets_per_point,
                self.metric,
                self.random_state,
            )
        else:
            rpc = rpc_result
        return ("rpc", rpc)

    def __str__(self):
        return (
            "Random Pair Correlation "
            + f"num triplets per point: {self.n_triplets_per_point}"
        )
