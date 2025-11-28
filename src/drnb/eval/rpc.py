from dataclasses import dataclass

import numpy as np
import scipy.stats

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


def _random_pair_setup(
    X: np.ndarray,
    X_new: np.ndarray,
    triplets: np.ndarray | None = None,
    random_state: int | None = None,
    n_triplets_per_point: int = 5,
    X_dist: np.ndarray | None = None,
    metric: str = "euclidean",
    Xnew_dist: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    X: np.ndarray,
    X_new: np.ndarray,
    triplets: np.ndarray | None = None,
    random_state: int | None = None,
    n_triplets_per_point: int = 5,
    return_triplets: bool = False,
    X_dist: np.ndarray | None = None,
    metric: str = "euclidean",
    Xnew_dist: np.ndarray | None = None,
) -> float | tuple[float, np.ndarray, np.ndarray]:
    """Evaluate the correlation between distances in the original and new embeddings
    using random triplets. Return the Pearson correlation coefficient. If
    return_triplets is True, also return the triplets and the distances in the original
    embedding."""
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

    correl = float(
        scipy.stats.pearsonr(X_dist.flatten(), Xnew_dist.flatten()).statistic
    )
    if return_triplets:
        return correl, triplets, X_dist
    return correl


def random_pairv(
    X: np.ndarray,
    X_new: np.ndarray,
    triplets: np.ndarray | None = None,
    random_state: int | None = None,
    n_triplets_per_point: int = 5,
    X_dist: np.ndarray | None = None,
    metric: str = "euclidean",
    Xnew_dist: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the distances in the original and new embeddings using random triplets.
    Return the distances in the original and new embeddings."""
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
    """Evaluate the embedding using random triplets. Return the Pearson correlation
    coefficient between the distances in the original and new embeddings.

    Attributes:
    random_state: int - random seed
    n_triplets_per_point: int - number of triplets per point
    use_precomputed_triplets: bool - whether to use precomputed triplets
    metric: str - distance metric
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

    def evaluate(
        self, X: np.ndarray, coords: np.ndarray, ctx: EmbedContext | None = None
    ) -> EvalResult:
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
            info={"metric": self.metric, "ntpp": self.n_triplets_per_point},
            value=rpc_result,
        )

    def evaluatev(
        self, X: np.ndarray, coords: np.ndarray, ctx: EmbedContext | None = None
    ) -> list[np.ndarray]:
        """Evaluate the embedding using random triplets. Return the distances in the
        original and new embeddings."""
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
