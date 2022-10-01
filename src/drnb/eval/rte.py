from dataclasses import dataclass

import numpy as np

from drnb.distance import distance_function

from ..log import log
from .base import EmbeddingEval
from .triplets import (
    calc_distances,
    find_precomputed_triplets,
    get_triplets,
    validate_triplets,
)


@dataclass
class RandomTripletEval(EmbeddingEval):
    random_state: int = None
    n_triplets_per_point: int = 5
    use_precomputed_triplets: bool = True
    metric: str = "euclidean"

    def evaluate(self, X, coords, ctx=None):
        idx = None
        X_dist = None

        if self.use_precomputed_triplets and ctx is not None:
            idx, X_dist = find_precomputed_triplets(
                ctx, self.n_triplets_per_point, self.metric
            )
            if idx is None:
                log.info("No precomputed triplets found")

        rte_result = random_triplet_eval(
            X,
            coords,
            random_state=self.random_state,
            n_triplets_per_point=self.n_triplets_per_point,
            triplets=idx,
            X_dist=X_dist,
            metric=self.metric,
        )
        return ("rte", rte_result)

    def __str__(self):
        return (
            "Random Triplet Evaluation "
            + f"num triplets per point: {self.n_triplets_per_point}"
        )


# Based on:
# https://github.com/YingfanWang/PaCMAP/blob/c7c45dbd0fec7736764d0e28203eb0e0515f3427/evaluation/evaluation.py
def random_triplet_evalv(
    X,
    X_new,
    triplets=None,
    random_state=None,
    n_triplets_per_point=5,
    return_triplets=False,
    X_dist=None,
    normalize=True,
    metric="euclidean",
):
    dist_fun = distance_function(metric)
    n_obs = X.shape[0]

    # Sampling Triplets
    # Five triplet per point
    if triplets is None:
        triplets = get_triplets(
            X, seed=random_state, n_triplets_per_point=n_triplets_per_point
        )
    else:
        validate_triplets(triplets, n_obs)
        n_triplets_per_point = triplets.shape[1]

    # Calculate the distances and generate labels
    if X_dist is None:
        X_dist = calc_distances(X, triplets, dist_fun)
    else:
        validate_triplets(X_dist, n_obs)

    labels = X_dist[:, :, 0] < X_dist[:, :, 1]

    # Calculate distances for LD
    Xnew_dist = calc_distances(X_new, triplets, dist_fun)
    pred_vals = Xnew_dist[:, :, 0] < Xnew_dist[:, :, 1]

    accv = np.sum(pred_vals == labels, axis=1, dtype=float)
    if normalize:
        accv /= float(n_triplets_per_point)
    if return_triplets:
        return accv, X_dist, n_triplets_per_point
    return accv


def random_triplet_eval(
    X,
    X_new,
    triplets=None,
    random_state=None,
    n_triplets_per_point=5,
    return_triplets=False,
    X_dist=None,
    normalize=True,
    metric="euclidean",
):
    res = random_triplet_evalv(
        X,
        X_new,
        triplets=triplets,
        random_state=random_state,
        n_triplets_per_point=n_triplets_per_point,
        return_triplets=return_triplets,
        X_dist=X_dist,
        normalize=normalize,
        metric=metric,
    )
    if return_triplets:
        accv = res[0]
    else:
        accv = res
    acc = np.mean(accv)

    if return_triplets:
        return acc, res[1], res[2]
    return acc
