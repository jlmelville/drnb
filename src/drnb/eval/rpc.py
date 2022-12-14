from dataclasses import dataclass

import scipy.stats

from drnb.distances import distance_function
from drnb.eval import EvalResult
from drnb.log import log

from ..triplets import (
    calc_distances,
    find_precomputed_triplets,
    get_triplets,
    validate_triplets,
)
from .base import EmbeddingEval


def _random_pair_setup(
    X,
    X_new,
    triplets=None,
    random_state=None,
    n_triplets_per_point=5,
    X_dist=None,
    metric="euclidean",
    Xnew_dist=None,
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

    if Xnew_dist is None:
        Xnew_dist = calc_distances(X_new, triplets, dist_fun)
    else:
        validate_triplets(Xnew_dist, n_obs)

    return (
        X_dist,
        Xnew_dist,
        triplets,
    )


def random_pair_correl_eval(
    X,
    X_new,
    triplets=None,
    random_state=None,
    n_triplets_per_point=5,
    return_triplets=False,
    X_dist=None,
    metric="euclidean",
    Xnew_dist=None,
):
    X_dist, Xnew_dist, triplets = _random_pair_setup(
        X,
        X_new,
        triplets=triplets,
        random_state=random_state,
        n_triplets_per_point=n_triplets_per_point,
        X_dist=X_dist,
        metric=metric,
        Xnew_dist=Xnew_dist,
    )

    correl = scipy.stats.pearsonr(X_dist.flatten(), Xnew_dist.flatten()).statistic
    if return_triplets:
        return correl, triplets, X_dist
    return correl


def random_pairv(
    X,
    X_new,
    triplets=None,
    random_state=None,
    n_triplets_per_point=5,
    X_dist=None,
    metric="euclidean",
    Xnew_dist=None,
):
    X_dist, Xnew_dist, triplets = _random_pair_setup(
        X,
        X_new,
        triplets=triplets,
        random_state=random_state,
        n_triplets_per_point=n_triplets_per_point,
        X_dist=X_dist,
        metric=metric,
        Xnew_dist=Xnew_dist,
    )

    return X_dist.flatten(), Xnew_dist.flatten()


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

    def _evaluate_setup(self, ctx):
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

    def evaluate(self, X, coords, ctx=None):
        idx, X_dist, Xnew_dist = self._evaluate_setup(ctx=ctx)

        rpc_result = random_pair_correl_eval(
            X,
            coords,
            random_state=self.random_state,
            triplets=idx,
            n_triplets_per_point=self.n_triplets_per_point,
            X_dist=X_dist,
            metric=self.metric,
            Xnew_dist=Xnew_dist,
        )

        return EvalResult(
            eval_type="RPC",
            label=str(self),
            info=dict(metric=self.metric, ntpp=self.n_triplets_per_point),
            value=rpc_result,
        )

    def evaluatev(self, X, coords, ctx=None):
        idx, X_dist, Xnew_dist = self._evaluate_setup(ctx=ctx)

        return random_pairv(
            X,
            coords,
            random_state=self.random_state,
            triplets=idx,
            n_triplets_per_point=self.n_triplets_per_point,
            X_dist=X_dist,
            metric=self.metric,
            Xnew_dist=Xnew_dist,
        )

    def __str__(self):
        return f"rpc-{self.n_triplets_per_point}-{self.metric}"
