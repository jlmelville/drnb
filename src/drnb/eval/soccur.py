from dataclasses import dataclass

import numpy as np
import scipy

from drnb.eval import EvalResult
from drnb.eval.nbrpres import get_xy_nbr_idxs
from drnb.neighbors import read_neighbors
from drnb.neighbors.hubness import s_occurrences
from drnb.util import islisty

from .base import EmbeddingEval


def soccur(
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
    result = []
    for nbrs in n_nbrs:
        if nbrs <= max_n_nbrs:
            sox = s_occurrences(x_nbrs[:, :nbrs], n_neighbors=nbrs)
            soy = s_occurrences(y_nbrs[:, :nbrs], n_neighbors=nbrs)
            correl = scipy.stats.pearsonr(sox, soy).statistic

            # result.append(np.mean(sox - soy))
            result.append(correl)
        else:
            result.append(np.nan)
    return result


@dataclass
class SOccurrenceEval(EmbeddingEval):
    use_precomputed_neighbors: bool = True
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
            kwargs = dict(
                drnb_home=ctx.drnb_home, sub_dir=ctx.nn_sub_dir, name=ctx.dataset_name
            )
        else:
            kwargs = {}

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

        return x_nbrs, y_nbrs, kwargs

    # def evaluatev(self, X, coords, ctx=None):
    #     x_nbrs, y_nbrs, nnp_kwargs = self._evaluate_setup(ctx)

    #     return nbr_presv(
    #         X,
    #         coords,
    #         n_nbrs=self.n_neighbors,
    #         x_metric=self.metric,
    #         include_self=self.include_self,
    #         verbose=self.verbose,
    #         x_nbrs=x_nbrs,
    #         y_nbrs=y_nbrs,
    #         **nnp_kwargs,
    #     )

    def evaluate(self, X, coords, ctx=None):
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
                info=dict(metric=self.metric, n_neighbors=n_nbrs),
                value=nnp,
            )
            for n_nbrs, nnp in zip(self.n_neighbors, results)
        ]

    def to_str(self, n_neighbors):
        include_self_str = "self" if self.include_self else "noself"
        return f"so-{n_neighbors}-{include_self_str}-{self.metric}"

    def __str__(self):
        include_self_str = "self" if self.include_self else "noself"
        return f"so-{self.n_neighbors}-{include_self_str}-{self.metric}"
