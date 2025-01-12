from dataclasses import dataclass
from typing import List

import numpy as np

from drnb.distances import distance_function
from drnb.embed.context import EmbedContext
from drnb.eval.base import EmbeddingEval, EvalResult
from drnb.log import log

from ..triplets import (
    calc_distances,
    find_precomputed_triplets,
    get_triplets,
    validate_triplets,
)


@dataclass
class RandomTripletEval(EmbeddingEval):
    """Random Triplet Evaluation (RTE) score. The RTE score is the fraction of
    triplets that are correctly ordered in the new embedding. A triplet is correctly
    ordered if the distance between the anchor and the positive point is smaller than
    the distance between the anchor and the negative point.

    Attributes:
        random_state: Random seed for sampling triplets.
        n_triplets_per_point: Number of triplets to sample per point.
        use_precomputed_triplets: Whether to use precomputed triplets.
        metric: Distance metric to use for calculating the RTE score.
    """

    random_state: int | None = None
    n_triplets_per_point: int = 5
    use_precomputed_triplets: bool = True
    metric: str = "euclidean"

    def requires(self):
        return {
            "name": "triplets",
            "n_triplets_per_point": self.n_triplets_per_point,
            "metric": self.metric,
            "random_state": self.random_state,
        }

    def _evaluate_setup(self, ctx=None):
        idx = None
        X_dist = None
        Xnew_dist = None

        if self.use_precomputed_triplets and ctx is not None:
            idx, X_dist = find_precomputed_triplets(
                dataset_name=ctx.dataset_name,
                triplet_sub_dir=ctx.triplet_sub_dir,
                n_triplets_per_point=self.n_triplets_per_point,
                metric=self.metric,
                drnb_home=ctx.drnb_home,
            )
            if idx is None:
                log.info("No precomputed triplets found")

            _, Xnew_dist = find_precomputed_triplets(
                dataset_name=ctx.embed_triplets_name,
                triplet_sub_dir=ctx.experiment_name,
                n_triplets_per_point=self.n_triplets_per_point,
                metric=self.metric,
                drnb_home=ctx.drnb_home,
            )
        return idx, X_dist, Xnew_dist

    def evaluate(
        self, X: np.ndarray, coords: np.ndarray, ctx: EmbedContext | None = None
    ) -> EvalResult:
        idx, X_dist, Xnew_dist = self._evaluate_setup(ctx=ctx)

        rte_result = random_triplet_eval(
            X,
            coords,
            random_state=self.random_state,
            n_triplets_per_point=self.n_triplets_per_point,
            triplets=idx,
            X_dist=X_dist,
            metric=self.metric,
            Xnew_dist=Xnew_dist,
        )

        return EvalResult(
            eval_type="RTE",
            label=str(self),
            info={"metric": self.metric, "ntpp": self.n_triplets_per_point},
            value=rte_result,
        )

    def evaluatev(
        self, X: np.ndarray, coords: np.ndarray, ctx: EmbedContext | None = None
    ) -> List[np.ndarray]:
        """Evaluate the Random Triplet Evaluation (RTE) score for each point in the
        dataset. The RTE score is the fraction of triplets that are correctly ordered
        in the new embedding. A triplet is correctly ordered if the distance between
        the anchor and the positive point is smaller than the distance between the
        anchor and the negative point."""
        idx, X_dist, Xnew_dist = self._evaluate_setup(ctx=ctx)

        return random_triplet_evalv(
            X,
            coords,
            random_state=self.random_state,
            n_triplets_per_point=self.n_triplets_per_point,
            triplets=idx,
            X_dist=X_dist,
            metric=self.metric,
            Xnew_dist=Xnew_dist,
        )

    def __str__(self):
        return f"rte-{self.n_triplets_per_point}-{self.metric}"


# Based on:
# https://github.com/YingfanWang/PaCMAP/blob/c7c45dbd0fec7736764d0e28203eb0e0515f3427/evaluation/evaluation.py
def random_triplet_evalv(
    X: np.ndarray,
    X_new: np.ndarray,
    triplets: np.ndarray | None = None,
    random_state: int | None = None,
    n_triplets_per_point: int = 5,
    return_triplets: bool = False,
    X_dist: np.ndarray | None = None,
    normalize: bool = True,
    metric: str = "euclidean",
    Xnew_dist: np.ndarray | None = None,
) -> tuple[float, np.ndarray, int] | float:
    """Calculate the Random Triplet Evaluation (RTE) score. The RTE score is the
    fraction of triplets that are correctly ordered in the new embedding. A triplet
    is correctly ordered if the distance between the anchor and the positive point
    is smaller than the distance between the anchor and the negative point.
    If return_triplets is True, the function will return the accuracy for each
    triplet and the number of triplets per point.
    If normalize is True, the accuracy will be normalized by the number of triplets
    per point.
    """
    dist_fun = distance_function(metric)
    n_obs = X.shape[0]

    # Sampling Triplets
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
    if Xnew_dist is None:
        Xnew_dist = calc_distances(X_new, triplets, dist_fun)
    else:
        validate_triplets(Xnew_dist, n_obs)
    pred_vals = Xnew_dist[:, :, 0] < Xnew_dist[:, :, 1]

    accv = np.sum(pred_vals == labels, axis=1, dtype=float)
    if normalize:
        accv /= float(n_triplets_per_point)
    if return_triplets:
        return accv, X_dist, n_triplets_per_point
    return accv


def random_triplet_eval(
    X: np.ndarray,
    X_new: np.ndarray,
    triplets: np.ndarray | None = None,
    random_state: int | None = None,
    n_triplets_per_point: int = 5,
    return_triplets: bool = False,
    X_dist: np.ndarray | None = None,
    normalize: bool = True,
    metric: str = "euclidean",
    Xnew_dist: np.ndarray | None = None,
) -> tuple[float, np.ndarray, int] | float:
    """Calculate the mean Random Triplet Evaluation (RTE) score over all points in the
    dataset. The RTE score is the fraction of triplets that are correctly ordered in
    in the new embedding. A triplet is correctly ordered if the distance between the
    anchor and the positive point is smaller than the distance between the anchor and
    the negative point.
    If return_triplets is True, the function will additionally return the accuracy for
    each triplet and the number of triplets per point.
    If normalize is True, the accuracy will be normalized by the number of triplets
    per point.
    """
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
        Xnew_dist=Xnew_dist,
    )
    if return_triplets:
        accv = res[0]
    else:
        accv = res
    acc = np.mean(accv)

    if return_triplets:
        return acc, res[1], res[2]
    return acc
