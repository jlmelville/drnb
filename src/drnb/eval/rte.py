from dataclasses import dataclass

import numpy as np

from .base import EmbeddingEval
from .triplets import calc_labels, get_triplets, validate_triplets


@dataclass
class RandomTripletEval(EmbeddingEval):
    random_state: int = None
    n_triplets_per_point: int = 5

    def evaluate(self, X, coords, ctx=None):
        rte = random_triplet_eval(
            X,
            coords,
            random_state=self.random_state,
            n_triplets_per_point=self.n_triplets_per_point,
        )
        return ("rte", rte)

    def __str__(self):
        return (
            "Random Triplet Evaluation "
            + f"num triplets per point: {self.n_triplets_per_point}"
        )


# https://github.com/YingfanWang/PaCMAP/blob/c7c45dbd0fec7736764d0e28203eb0e0515f3427/evaluation/evaluation.py
def random_triplet_eval(
    X, X_new, triplets=None, random_state=None, n_triplets_per_point=5
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
    # broadcasting to b flattens the anchors and triplets to generate a list of pairs:
    # [0, triplet00], [0, triplet01], [0, triplet10], [0, triplet11] etc.
    b = np.broadcast(anchors, triplets)
    labels = calc_labels(X, b)

    # Calculate distances for LD
    b = np.broadcast(anchors, triplets)
    pred_vals = calc_labels(X_new, b)

    correct = np.sum(pred_vals == labels)
    acc = correct / n_obs / n_triplets_per_point
    return acc
