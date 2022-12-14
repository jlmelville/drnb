from dataclasses import dataclass

import numpy as np

from drnb.eval import EvalResult
from drnb.log import log
from drnb.neighbors import calculate_neighbors, get_neighbors, read_neighbors
from drnb.util import islisty

from .base import EmbeddingEval


def nn_accv(approx_indices, true_indices, normalize=True):
    result = np.zeros(approx_indices.shape[0])
    for i in range(approx_indices.shape[0]):
        n_correct = np.intersect1d(approx_indices[i], true_indices[i]).shape[0]
        if normalize:
            result[i] = n_correct / true_indices.shape[1]
    return result


def nn_acc(approx_indices, true_indices):
    return np.mean(nn_accv(approx_indices, true_indices))


def get_xy_nbr_idxs(
    X,
    Y,
    n_nbrs=15,
    x_method="exact",
    x_metric="euclidean",
    x_method_kwds=None,
    y_method="exact",
    y_metric="euclidean",
    y_method_kwds=None,
    include_self=False,
    verbose=False,
    x_nbrs=None,
    y_nbrs=None,
    name=None,
    drnb_home=None,
    sub_dir="nn",
):
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
    X,
    Y,
    n_nbrs=15,
    x_method="exact",
    x_metric="euclidean",
    x_method_kwds=None,
    y_method="exact",
    y_metric="euclidean",
    y_method_kwds=None,
    include_self=False,
    verbose=False,
    x_nbrs=None,
    y_nbrs=None,
    name=None,
    drnb_home=None,
    sub_dir="nn",
):
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
    X,
    Y,
    n_nbrs=15,
    x_method="exact",
    x_metric="euclidean",
    x_method_kwds=None,
    y_method="exact",
    y_metric="euclidean",
    y_method_kwds=None,
    include_self=False,
    normalize=True,
    verbose=False,
    x_nbrs=None,
    y_nbrs=None,
    name=None,
    drnb_home=None,
    sub_dir="nn",
):
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
    use_precomputed_neighbors: bool = True
    # this translates to the x-metric in the nbr_pres
    metric: str = "euclidean"
    n_neighbors: int = 15  # can also be a list
    include_self: bool = False
    verbose: bool = False

    def listify_n_neighbors(self):
        if not islisty(self.n_neighbors):
            self.n_neighbors = [self.n_neighbors]

    def requires(self):
        self.listify_n_neighbors()
        return dict(
            name="neighbors",
            metric=self.metric,
            n_neighbors=int(np.max(self.n_neighbors)),
        )

    def _evaluate_setup(self, ctx=None):
        self.listify_n_neighbors()

        if ctx is not None:
            nnp_kwargs = dict(
                drnb_home=ctx.drnb_home, sub_dir=ctx.nn_sub_dir, name=ctx.dataset_name
            )
        else:
            nnp_kwargs = {}

        x_nbrs = None
        y_nbrs = None
        if self.use_precomputed_neighbors and ctx is not None:
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

    def evaluatev(self, X, coords, ctx=None):
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

    def evaluate(self, X, coords, ctx=None):
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
                info=dict(metric=self.metric, n_neighbors=n_nbrs),
                value=nnp,
            )
            for n_nbrs, nnp in zip(self.n_neighbors, nnps)
        ]

    def to_str(self, n_neighbors):
        include_self_str = "self" if self.include_self else "noself"
        return f"nnp-{n_neighbors}-{include_self_str}-{self.metric}"

    def __str__(self):
        include_self_str = "self" if self.include_self else "noself"
        return f"nnp-{self.n_neighbors}-{include_self_str}-{self.metric}"
