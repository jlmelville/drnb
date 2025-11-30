from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drnb.embed.context import EmbedContext
from drnb.eval.base import EmbeddingEval, EvalResult
from drnb.neighbors.hubness import nn_to_sparse
from drnb.neighbors.nbrinfo import NearestNeighbors
from drnb.neighbors.store import read_neighbors

from .nbrpres import get_xy_nbr_idxs


def unn_accv(
    approx_indices: np.ndarray | tuple | NearestNeighbors,
    true_indices: np.ndarray | tuple | NearestNeighbors,
) -> np.ndarray:
    """Calculate the undirected nearest neighbor accuracy for each point in the dataset.
    The accuracy is the Jaccard similarity between the sets of neighbors of the point in
    the approximate and true nearest neighbor indices. Neighbors are symmetrized before
    calculating the Jaccard similarity."""
    amat = nn_to_sparse(approx_indices, symmetrize="or")
    amat.eliminate_zeros()
    aptr = amat.tocsr().indptr
    aidx = amat.col

    tmat = nn_to_sparse(true_indices, symmetrize="or")
    tmat.eliminate_zeros()
    tptr = tmat.tocsr().indptr
    tidx = tmat.col

    result = np.zeros(approx_indices.shape[0])
    for i in range(approx_indices.shape[0]):
        aidxi = set(list(aidx[aptr[i] : aptr[i + 1]]))
        tidxi = set(list(tidx[tptr[i] : tptr[i + 1]]))

        result[i] = float(len(aidxi.intersection(tidxi)) / len(aidxi.union(tidxi)))
    return result


def unn_acc(
    approx_indices: np.ndarray | tuple | NearestNeighbors,
    true_indices: np.ndarray | tuple | NearestNeighbors,
) -> float:
    """Calculate the mean undirected nearest neighbor accuracy for the dataset. The
    accuracy is the mean Jaccard similarity between the sets of neighbors of each point
    in the approximate and true nearest neighbor indices. Neighbors are symmetrized
    before calculating the Jaccard similarity."""
    return np.mean(unn_accv(approx_indices, true_indices))


def unbr_pres(
    X: np.ndarray,
    Y: np.ndarray,
    n_nbrs: int | list[int] = 15,
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
) -> list[float]:
    """Calculate the undirected nearest neighbor preservation for the dataset: the mean
    overlap of the symmetrized nearest neighbors of X and those of Y. The number of
    neighbors to consider can be a single integer or a list of integers. The accuracy
    is calculated for each value in the list and the results are returned in a list.

    If the number of neighbors requested is larger than the number of neighbors that
    can be calculated, NaN is returned for that value.

    If include_self is True, the first column of the indices will be the index of the
    item itself: this is normal for the definition of nearest neighbor, but reduces
    the useful range of the metric.

    If there are no precomputed neighbors available, they will be calculated and saved.
    """
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

    # contents of n_nbrs may still be larger than the number of neighbors that can be
    # sensibly calculated so check and use NaN for those cases
    unn_accs = []
    for nbrs in n_nbrs:
        if nbrs <= max_n_nbrs:
            unn_accs.append(
                unn_acc(
                    approx_indices=y_nbrs[:, :nbrs],
                    true_indices=x_nbrs[:, :nbrs],
                )
            )
        else:
            unn_accs.append(np.nan)
    return unn_accs


def unbr_presv(
    X: np.ndarray,
    Y: np.ndarray,
    n_nbrs: int | list[int] = 15,
    x_nbrs: NearestNeighbors | None = None,
    y_nbrs: NearestNeighbors | None = None,
    x_method: str = "exact",
    x_metric: str = "euclidean",
    x_method_kwds: dict | None = None,
    y_method: str = "exact",
    y_metric: str = "euclidean",
    y_method_kwds: dict | None = None,
    include_self: bool = False,
    verbose: bool = False,
    name: str | None = None,
    drnb_home: Path | str | None = None,
    sub_dir: str = "nn",
) -> list[np.ndarray]:
    """Calculate the per-item undirected nearest neighbor preservation for the dataset:
    the overlap of the symmetrized nearest neighbors of X and those of Y. The number of
    neighbors to consider can be a single integer or a list of integers. The accuracy
    is calculated for each value in the list and the results are returned in a list.

    If the number of neighbors requested is larger than the number of neighbors that
    can be calculated, NaN is returned for that value.

    If include_self is True, the first column of the indices will be the index of the
    item itself: this is normal for the definition of nearest neighbor, but reduces the
    useful range of the metric.

    If there are no precomputed neighbors available, they will be calculated and saved.
    """
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

    # contents of n_nbrs may still be larger than the number of neighbors that can be
    # sensibly calculated so check and use NaN for those cases
    unn_accvs = []
    for nbrs in n_nbrs:
        if nbrs <= max_n_nbrs:
            unn_accvs.append(unn_accv(approx_indices=y_nbrs, true_indices=x_nbrs))
        else:
            unn_accvs.append([])
    return unn_accvs


@dataclass
class UndirectedNbrPreservationEval(EmbeddingEval):
    """Compute the undirected nearest neighbor preservation of an embedding: the overlap
    of the symmetrized nearest neighbors of the original and embedded points.

    Attributes:
    use_precomputed_knn: bool - use precomputed neighbors if available
    metric: str - distance metric to use (default: "euclidean")
    n_neighbors: int | List[int] - number of neighbors to consider (default: 15)
    include_self: bool - include the item itself in the neighbors (default: False)
    verbose: bool - print progress information (default: False)
    """

    use_precomputed_knn: bool = True
    # this translates to the x-metric in the nbr_pres
    metric: str = "euclidean"
    n_neighbors: int | list[int] = 15  # can also be a list
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
    ) -> tuple[NearestNeighbors | None, NearestNeighbors | None, dict]:
        self._listify_n_neighbors()

        if ctx is not None:
            nnp_kwargs = {
                "drnb_home": ctx.drnb_home,
                "sub_dir": ctx.nn_sub_dir,
                "name": ctx.dataset_name,
            }
        else:
            nnp_kwargs = {}

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

        return x_nbrs, y_nbrs, nnp_kwargs

    def evaluatev(
        self, X: np.ndarray, coords: np.ndarray, ctx: EmbedContext | None = None
    ) -> list[np.ndarray]:
        """Evaluate the per-item nearest neighbor preservation. Return a list of arrays
        of accuracies, one per value of n_neighbors in the evaluation."""
        x_nbrs, y_nbrs, nnp_kwargs = self._evaluate_setup(ctx)

        return unbr_presv(
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

    def evaluate(
        self, X: np.ndarray, coords: np.ndarray, ctx: EmbedContext | None = None
    ) -> EvalResult:
        x_nbrs, y_nbrs, nnp_kwargs = self._evaluate_setup(ctx)

        nnps = unbr_pres(
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
                eval_type="UNNP",
                label=self.to_str(n_nbrs),
                info={"metric": self.metric, "n_neighbors": n_nbrs},
                value=nnp,
            )
            for n_nbrs, nnp in zip(self.n_neighbors, nnps)
        ]

    def to_str(self, n_neighbors: int) -> str:
        """Create a string representation of the evaluation for a given number of
        neighbors."""
        include_self_str = "self" if self.include_self else "noself"
        return f"unnp-{n_neighbors}-{include_self_str}-{self.metric}"

    def __str__(self):
        include_self_str = "self" if self.include_self else "noself"
        return f"unnp-{self.n_neighbors}-{include_self_str}-{self.metric}"
