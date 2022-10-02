from dataclasses import dataclass

import scipy.stats

from drnb.distance import distance_function
from drnb.eval import EvalResult
from drnb.log import log

from ..triplets import (
    calc_distances,
    find_precomputed_triplets,
    get_triplets,
    validate_triplets,
)
from .base import EmbeddingEval


def random_pair_correl_eval(
    X,
    X_new,
    triplets=None,
    random_state=None,
    n_triplets_per_point=5,
    return_triplets=False,
    X_dist=None,
    metric="euclidean",
):
    dist_fun = distance_function(metric)

    n_obs = X.shape[0]
    if triplets is None:
        triplets = get_triplets(
            X, seed=random_state, n_triplets_per_point=n_triplets_per_point
        )
    else:
        validate_triplets(triplets, n_obs)
        n_triplets_per_point = triplets.shape[1]

    if X_dist is None:
        X_dist = calc_distances(X, triplets, dist_fun)
    else:
        validate_triplets(X_dist, n_obs)

    Xnew_dist = calc_distances(X_new, triplets, dist_fun)

    correl = scipy.stats.pearsonr(X_dist.flatten(), Xnew_dist.flatten()).statistic
    if return_triplets:
        return correl, triplets, X_dist
    return correl


@dataclass
class RandomPairCorrelEval(EmbeddingEval):
    random_state: int = None
    n_triplets_per_point: int = 5
    use_precomputed_triplets: bool = True
    metric: str = "euclidean"

    def requires(self):
        return dict(
            name="triplets",
            n_triplets_per_point=self.n_triplets_per_point,
            metric=self.metric,
            random_state=self.random_state,
        )

    def evaluate(self, X, coords, ctx=None):
        idx = None
        X_dist = None

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

        rpc_result = random_pair_correl_eval(
            X,
            coords,
            random_state=self.random_state,
            triplets=idx,
            n_triplets_per_point=self.n_triplets_per_point,
            X_dist=X_dist,
            metric=self.metric,
        )

        return EvalResult(
            eval_type="RPC",
            label=str(self),
            info=dict(metric=self.metric, ntpp=self.n_triplets_per_point),
            value=rpc_result,
        )

    def __str__(self):
        return f"rpc-{self.n_triplets_per_point}-{self.metric}"
