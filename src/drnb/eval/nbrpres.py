from dataclasses import dataclass

import numpy as np

from drnb.io import islisty
from drnb.neighbors import get_neighbors

from .base import EmbeddingEval


def nn_accv(approx_indices, true_indices):
    result = np.zeros(approx_indices.shape[0])
    for i in range(approx_indices.shape[0]):
        n_correct = np.intersect1d(approx_indices[i], true_indices[i]).shape[0]
        result[i] = n_correct / true_indices.shape[1]
    return result


def nn_acc(approx_indices, true_indices):
    return np.mean(nn_accv(approx_indices, true_indices))


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
    verbose=False,
    x_nbrs=None,
    y_nbrs=None,
):
    if isinstance(n_nbrs, int):
        n_nbrs = [n_nbrs]
    max_n_nbrs = int(np.max(n_nbrs))

    calc_x_nbrs = True
    if x_nbrs is not None:
        calc_x_nbrs = max_n_nbrs > x_nbrs.shape[1]

    if calc_x_nbrs:
        x_nbrs = get_neighbors(
            X,
            metric=x_metric,
            n_neighbors=max_n_nbrs,
            method=x_method,
            return_distance=False,
            method_kwds=x_method_kwds,
            verbose=verbose,
        )

    calc_y_nbrs = True
    if y_nbrs is not None:
        calc_y_nbrs = max_n_nbrs > y_nbrs.shape[1]

    if calc_y_nbrs:
        y_nbrs = get_neighbors(
            Y,
            metric=y_metric,
            n_neighbors=max_n_nbrs,
            method=y_method,
            return_distance=False,
            method_kwds=y_method_kwds,
            verbose=verbose,
        )

    return [
        nn_acc(approx_indices=y_nbrs[:, :nbrs], true_indices=x_nbrs[:, :nbrs])
        for nbrs in n_nbrs
    ]


@dataclass
class NbrPreservationEval(EmbeddingEval):
    n_neighbors: int = 15  # can also be a list
    verbose: bool = False

    def evaluate(self, X, coords):
        nnps = nbr_pres(X, coords, n_nbrs=self.n_neighbors, verbose=self.verbose)
        if not islisty(self.n_neighbors):
            self.n_neighbors = [self.n_neighbors]
        return [(f"nnp{n_nbrs}", nnp) for n_nbrs, nnp in zip(self.n_neighbors, nnps)]