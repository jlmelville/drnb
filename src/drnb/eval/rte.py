from dataclasses import dataclass

import numpy as np

from ..log import log
from .base import EmbeddingEval
from .triplets import (
    cache_triplets,
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

        rte_result = random_triplet_eval(
            X,
            coords,
            random_state=self.random_state,
            n_triplets_per_point=self.n_triplets_per_point,
            triplets=idx,
            X_dist=X_dist,
            return_triplets=return_triplets,
        )
        if return_triplets:
            rte = rte_result[0]
            idx = rte_result[1]
            X_dist = rte_result[2]
            cache_triplets(
                idx,
                X_dist,
                ctx,
                self.n_triplets_per_point,
                self.metric,
                self.random_state,
            )
        else:
            rte = rte_result
        return ("rte", rte)

    def __str__(self):
        return (
            "Random Triplet Evaluation "
            + f"num triplets per point: {self.n_triplets_per_point}"
        )


# https://github.com/YingfanWang/PaCMAP/blob/c7c45dbd0fec7736764d0e28203eb0e0515f3427/evaluation/evaluation.py
def random_triplet_eval(
    X,
    X_new,
    triplets=None,
    random_state=None,
    n_triplets_per_point=5,
    return_triplets=False,
    X_dist=None,
    normalize=True,
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
    )
    if return_triplets:
        accv = res[0]
    else:
        accv = res
    acc = np.mean(accv)

    if return_triplets:
        return acc, res[1], res[2]
    return acc


def random_triplet_evalv(
    X,
    X_new,
    triplets=None,
    random_state=None,
    n_triplets_per_point=5,
    return_triplets=False,
    X_dist=None,
    normalize=True,
):
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

    # 3D (Nx1x1) array where e.g. anchors[i][0][0] = i: [[[0]], [[1]], [[2]] ... [[n_obs]]]
    anchors = np.arange(n_obs).reshape((-1, 1, 1))

    # Calculate the distances and generate labels
    # broadcasting to bpairs flattens the anchors and triplets to generate a list of
    # pairs:
    # [0, triplet00], [0, triplet01], [0, triplet10], [0, triplet11] etc.
    if X_dist is None:
        bpairs = np.broadcast(anchors, triplets)
        X_dist = calc_distances(X, bpairs)
    else:
        validate_triplets(X_dist, n_obs)

    labels = X_dist[:, :, 0] < X_dist[:, :, 1]

    # Calculate distances for LD
    bpairs = np.broadcast(anchors, triplets)
    X_new_dist = calc_distances(X_new, bpairs)
    pred_vals = X_new_dist[:, :, 0] < X_new_dist[:, :, 1]

    accv = np.sum(pred_vals == labels, axis=1, dtype=float)
    if normalize:
        accv /= float(n_triplets_per_point)
    if return_triplets:
        return accv, X_dist, n_triplets_per_point
    return accv
