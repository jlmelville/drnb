from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import scipy

from drnb.embed.context import EmbedContext
from drnb.eval.base import EmbeddingEval, EvalResult
from drnb.eval.nbrpres import get_xy_nbr_idxs
from drnb.neighbors.nbrinfo import NearestNeighbors
from drnb.neighbors.store import read_neighbors
from drnb.neighbors.hubness import s_occurrences


def soccur(
    X: np.ndarray,
    Y: np.ndarray,
    n_nbrs: int | List[int] = 15,
    x_nbrs: NearestNeighbors | None = None,
    y_nbrs: NearestNeighbors | None = None,
    include_self: bool = False,
    x_method: str = "exact",
    x_metric: str = "euclidean",
    x_method_kwds: dict | None = None,
    y_method: str = "exact",
    y_metric: str = "euclidean",
    y_method_kwds: dict | None = None,
    verbose: bool = False,
    name: str | None = None,
    drnb_home: Path | str | None = None,
    sub_dir: str = "nn",
) -> List[float]:
    """Calculate the correlation of the s-occurrences of each point in the
    nearest neighbors of X and Y for each value of n_nbrs.
    The s-occurrence of a point is the number of mutual nearest neighbors."""
    x_nbrs, y_nbrs = get_xy_nbr_idxs(
        X,
        Y,
        n_nbrs=n_nbrs,
        x_method=x_method,
        x_metric=x_metric,
        x_method_kwds=x_method_kwds,
        y_method=y_method,
        y_metric=y_metric,
        y_method_kwds=y_method_kwds,
        include_self=include_self,
        verbose=verbose,
        x_nbrs=x_nbrs,
        y_nbrs=y_nbrs,
        name=name,
        drnb_home=drnb_home,
        sub_dir=sub_dir,
    )

    if isinstance(n_nbrs, int):
        n_nbrs = [n_nbrs]

    max_n_nbrs = x_nbrs.shape[1]
    result = []
    for nbrs in n_nbrs:
        if nbrs <= max_n_nbrs:
            sox = s_occurrences(x_nbrs[:, :nbrs], n_neighbors=nbrs)
            soy = s_occurrences(y_nbrs[:, :nbrs], n_neighbors=nbrs)
            correl = scipy.stats.pearsonr(sox, soy).statistic

            result.append(correl)
        else:
            result.append(np.nan)
    return result


@dataclass
class SOccurrenceEval(EmbeddingEval):
    """Evaluate the embedding using the s-occurrence metric. Return the Pearson
    correlation coefficient between the s-occurrences of the nearest neighbors of the
    original and new embeddings.

    Attributes:
        use_precomputed_knn: Whether to use precomputed neighbors.
        metric: Distance metric.
        n_neighbors: Number of neighbors to use for the evaluation. Can also be a list.
        include_self: Whether to include the point itself in the neighbors.
        verbose: Whether to print verbose output.
    """

    use_precomputed_knn: bool = True
    metric: str = "euclidean"
    n_neighbors: int = 15  # can also be a list
    include_self: bool = False
    verbose: bool = False

    def _listify_n_neighbors(self):
        if not isinstance(self.n_neighbors, (list, tuple)):
            self.n_neighbors = [self.n_neighbors]

    def requires(self):
        self._listify_n_neighbors()
        return {
            "name": "neighbors",
            "metric": self.metric,
            "n_neighbors": int(np.max(self.n_neighbors)),
        }

    def _evaluate_setup(
        self, ctx: EmbedContext | None = None
    ) -> Tuple[NearestNeighbors | None, NearestNeighbors | None, dict]:
        self._listify_n_neighbors()

        if ctx is not None:
            kwargs = {
                "drnb_home": ctx.drnb_home,
                "sub_dir": ctx.nn_sub_dir,
                "name": ctx.dataset_name,
            }
        else:
            kwargs = {}

        x_nbrs = None
        y_nbrs = None
        if self.use_precomputed_knn and ctx is not None:
            n_nbrs = int(np.max(self.n_neighbors))
            if not self.include_self:
                n_nbrs += 1
            x_nbrs = read_neighbors(
                name=ctx.dataset_name,
                n_neighbors=n_nbrs,
                metric=self.metric,
                exact=True,
                drnb_home=ctx.drnb_home,
                sub_dir=ctx.nn_sub_dir,
                return_distance=True,
            )
            y_nbrs = read_neighbors(
                name=ctx.embed_nn_name,
                n_neighbors=n_nbrs,
                metric=self.metric,
                exact=True,
                drnb_home=ctx.drnb_home,
                sub_dir=ctx.experiment_name,
                return_distance=True,
            )

        return x_nbrs, y_nbrs, kwargs

    def evaluate(
        self, X: np.ndarray, coords: np.ndarray, ctx: EmbedContext | None = None
    ) -> EvalResult:
        x_nbrs, y_nbrs, nnp_kwargs = self._evaluate_setup(ctx)

        results = soccur(
            X,
            coords,
            n_nbrs=self.n_neighbors,
            x_metric=self.metric,
            include_self=self.include_self,
            verbose=self.verbose,
            x_nbrs=x_nbrs,
            y_nbrs=y_nbrs,
            **nnp_kwargs,
        )

        return [
            EvalResult(
                eval_type="SOccur",
                label=self.to_str(n_nbrs),
                info={"metric": self.metric, "n_neighbors": n_nbrs},
                value=nnp,
            )
            for n_nbrs, nnp in zip(self.n_neighbors, results)
        ]

    def to_str(self, n_neighbors) -> str:
        """Create a string representation of the evaluation for a given number of
        neighbors."""
        include_self_str = "self" if self.include_self else "noself"
        return f"so-{n_neighbors}-{include_self_str}-{self.metric}"

    def __str__(self):
        include_self_str = "self" if self.include_self else "noself"
        return f"so-{self.n_neighbors}-{include_self_str}-{self.metric}"
