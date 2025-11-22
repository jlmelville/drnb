from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from drnb.embed.context import EmbedContext
from drnb.eval.base import EmbeddingEval, EvalResult
from drnb.log import log
from drnb.neighbors import (
    NearestNeighbors,
    calculate_neighbors,
    get_neighbors,
    read_neighbors,
)


def nn_accv(
    approx_indices: np.ndarray, true_indices: np.ndarray, normalize: bool = True
) -> np.ndarray:
    """Calculate the nearest neighbor accuracy for each point in the dataset, the number
    of neighbors that overlap between the true and approximate nearest neighbors.
    Returns a 1D array of accuracies, one per item in the dataset. If normalize is True,
    the accuracy is normalized by the number of neighbors."""
    result = np.zeros(approx_indices.shape[0])
    for i in range(approx_indices.shape[0]):
        n_correct = np.intersect1d(approx_indices[i], true_indices[i]).shape[0]
        if normalize:
            result[i] = n_correct / true_indices.shape[1]
    return result


def nn_acc(approx_indices: np.ndarray, true_indices: np.ndarray) -> float:
    """Calculate the mean nearest neighbor accuracy for the dataset."""
    return np.mean(nn_accv(approx_indices, true_indices))


def get_xy_nbr_idxs(
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
    fail_on_recalc: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the nearest neighbor indices for X and Y, calculating them if necessary.
    The returned indices will be of shape (n_items, max_n_nbrs) where max_n_nbrs is
    large enough to accommodate the largest value in n_nbrs. If include_self is True,
    the first column of the indices will be the index of the item itself. Note that in
    this case the number of neighbors requested will be increased by 1, which could
    trigger a recalculation of the neighbors. In the case of X-neighbors, this could be
    both slow and surprising, so by default this will raise an error. Set
    fail_on_recalc to False to allow the recalculation. It is expected that the Y
    neighbors will be recalculated in this case, so no error will be raised.
    """
    if isinstance(n_nbrs, int):
        n_nbrs = [n_nbrs]

    start_idx = 0
    if not include_self:
        n_nbrs = [n + 1 for n in n_nbrs]
        start_idx = 1

    max_n_nbrs = int(np.max(n_nbrs))

    n_items = Y.shape[0]
    if n_items < max_n_nbrs:
        log.warning(
            "%d nearest neighbors requested but only %d items are available",
            max_n_nbrs,
            n_items,
        )
        max_n_nbrs = n_items

    if verbose:
        log.info("Getting Y neighbors")
    calc_y_nbrs = True
    if y_nbrs is not None:
        calc_y_nbrs = max_n_nbrs > y_nbrs.idx.shape[1]

    if calc_y_nbrs:
        y_nbrs = calculate_neighbors(
            data=Y,
            n_neighbors=max_n_nbrs,
            metric=y_metric,
            method=y_method,
            return_distance=False,
            method_kwds=y_method_kwds,
            verbose=verbose,
        )

    if verbose:
        log.info("Getting X neighbors")
    calc_x_nbrs = True
    if x_nbrs is not None:
        calc_x_nbrs = max_n_nbrs > x_nbrs.idx.shape[1]

    if calc_x_nbrs:
        if fail_on_recalc:
            raise ValueError("Recalculation of X neighbors required")
        cache = name is not None
        x_nbrs = get_neighbors(
            data=X,
            n_neighbors=max_n_nbrs,
            metric=x_metric,
            method=x_method,
            return_distance=False,
            method_kwds=x_method_kwds,
            verbose=verbose,
            drnb_home=drnb_home,
            sub_dir=sub_dir,
            name=name,
            cache=cache,
        )

    return x_nbrs.idx[:, start_idx:max_n_nbrs], y_nbrs.idx[:, start_idx:max_n_nbrs]


def nbr_pres(
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
    """Calculate the nearest neighbor preservation for the dataset: the mean overlap
    of the nearest neighbors of X and those of Y. The number of neighbors to consider
    can be a single integer or a list of integers. The accuracy is calculated for each
    value in the list and the results are returned in a list.

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
    # sensibly calculated so check and use NaN for those cases. This is not an error
    # because we could have a mix of small and large datasets we are comparing.
    nn_accs = []
    for nbrs in n_nbrs:
        if nbrs <= max_n_nbrs:
            nn_accs.append(
                nn_acc(
                    approx_indices=y_nbrs[:, :nbrs],
                    true_indices=x_nbrs[:, :nbrs],
                )
            )
        else:
            nn_accs.append(np.nan)
    return nn_accs


def nbr_presv(
    X: np.ndarray,
    Y: np.ndarray,
    n_nbrs: int | List[int] = 15,
    x_nbrs: NearestNeighbors | None = None,
    y_nbrs: NearestNeighbors | None = None,
    normalize: bool = True,
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
) -> List[np.ndarray]:
    """Calculate the per-item nearest neighbor preservation for the dataset: the
    overlap of the nearest neighbors of X and those of Y. The number of neighbors to
    consider can be a single integer or a list of integers. The accuracy is calculated
    for each value in the list and the results are returned in a list.

    If the number of neighbors requested is larger than the number of neighbors that
    can be calculated, NaN is returned for that value.

    If include_self is True, the first column of the indices will be the index of the
    item itself: this is normal for the definition of nearest neighbor, but reduces the
    useful range of the metric.

    If normalize is True, the accuracy is normalized by the number of neighbors.

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
    nn_accvs = []
    for nbrs in n_nbrs:
        if nbrs <= max_n_nbrs:
            nn_accvs.append(
                nn_accv(approx_indices=y_nbrs, true_indices=x_nbrs, normalize=normalize)
            )
        else:
            nn_accvs.append([])
    return nn_accvs


@dataclass
class NbrPreservationEval(EmbeddingEval):
    """Evaluate the preservation of nearest neighbors in the embedding.

    Attributes:
    use_precomputed_knn: bool - use precomputed nearest neighbors if available
    metric: str - metric to use for nearest neighbor calculation
    n_neighbors: int | List[int] - number of neighbors to consider
    include_self: bool - include the item itself in the nearest neighbors
    verbose: bool - print verbose output
    """

    use_precomputed_knn: bool = True
    # this translates to the x-metric in the nbr_pres
    metric: str = "euclidean"
    n_neighbors: int = 15  # can also be a list
    include_self: bool = False
    verbose: bool = False

    def _listify_n_neighbors(self):
        """Ensure that n_neighbors is a list."""
        if not isinstance(self.n_neighbors, (list, tuple)):
            self.n_neighbors = [self.n_neighbors]

    def requires(self) -> dict:
        self._listify_n_neighbors()
        return {
            "name": "neighbors",
            "metric": self.metric,
            "n_neighbors": int(np.max(self.n_neighbors)),
        }

    def _evaluate_setup(
        self, ctx: EmbedContext | None = None
    ) -> Tuple[NearestNeighbors | None, NearestNeighbors | None, dict]:
        """Setup the evaluation of nearest neighbor preservation.

        Returns:
            x_nbrs: NearestNeighbors | None - precomputed X neighbors. Exact neighbors
              are used if available, otherwise approximate neighbors are used. If
              approximate neighbors are not available, then None is returned, but it
              is likely that this will lead to an error in the evaluation.
            y_nbrs: NearestNeighbors | None - precomputed Y neighbors. If exact
              neighbors are not available, then approximate neighbors are not searched
              for. Unlike with X neighbors, it is expected and not an error if
              neighbors are not available.
            nnp_kwargs: dict - keyword arguments for the evaluation


        """
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
            # exact=None will search for exact and then approximate neighbors of X
            x_nbrs = read_neighbors(
                name=ctx.dataset_name,
                n_neighbors=n_nbrs,
                metric=self.metric,
                exact=None,
                drnb_home=ctx.drnb_home,
                sub_dir=ctx.nn_sub_dir,
                return_distance=True,
            )
            # we only want exact neighbors for Y and we will recalculate downstream if
            # necessary
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
    ) -> List[np.ndarray]:
        """Evaluate the per-item nearest neighbor preservation. Return a list of arrays
        of accuracies, one per value of n_neighbors in the evaluation."""
        x_nbrs, y_nbrs, nnp_kwargs = self._evaluate_setup(ctx)

        return nbr_presv(
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

        nnps = nbr_pres(
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
                eval_type="NNP",
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
        return f"nnp-{n_neighbors}-{include_self_str}-{self.metric}"

    def __str__(self):
        include_self_str = "self" if self.include_self else "noself"
        return f"nnp-{self.n_neighbors}-{include_self_str}-{self.metric}"
