from dataclasses import dataclass
from typing import List, Union

import numpy as np
import pandas as pd
from numba import njit, prange

from drnb.eval import EvalResult
from drnb.eval.base import EmbeddingEval
from drnb.io.dataset import read_target
from drnb.log import log
from drnb.neighbors import calculate_neighbors


def label_pres(
    data,
    labels,
    n_nbrs: Union[List[int], int] = 15,
    method="exact",
    metric="euclidean",
    method_kwds=None,
    nbrs=None,
    verbose=False,
):
    if isinstance(n_nbrs, int):
        n_nbrs = [n_nbrs]

    if labels is None:
        log.warning("No labels provided")
        return [np.nan] * len(n_nbrs)

    max_n_nbrs = max(n_nbrs)
    nbr_idxs = get_nbr_idxs(
        data,
        n_nbrs=max_n_nbrs,
        method=method,
        metric=metric,
        method_kwds=method_kwds,
        nbrs=nbrs,
        verbose=verbose,
    )
    max_n_nbrs = nbr_idxs.shape[1]
    lps = []
    for nbrs in n_nbrs:
        if nbrs <= max_n_nbrs:
            predicted_labels = labels[nbr_idxs[:, :nbrs]]
            lps.append(np.mean(majority_vote(labels, predicted_labels)))
        else:
            lps.append(np.nan)
    return lps


def label_presv(
    data,
    labels,
    n_nbrs: Union[List[int], int] = 15,
    method="exact",
    metric="euclidean",
    method_kwds=None,
    nbrs=None,
    verbose=False,
):
    if isinstance(n_nbrs, int):
        n_nbrs = [n_nbrs]

    if labels is None:
        log.warning("No labels provided")
        return [] * len(n_nbrs)

    max_n_nbrs = max(n_nbrs)
    nbr_idxs = get_nbr_idxs(
        data,
        n_nbrs=max_n_nbrs,
        method=method,
        metric=metric,
        method_kwds=method_kwds,
        nbrs=nbrs,
        verbose=verbose,
    )
    max_n_nbrs = nbr_idxs.shape[1]
    lps = []
    for nbrs in n_nbrs:
        if nbrs <= max_n_nbrs:
            predicted_labels = labels[nbr_idxs[:, :nbrs]]
            lps.append(majority_vote(labels, predicted_labels))
        else:
            lps.append([])
    return lps


@njit(parallel=True)
def majority_vote(true_labels, predicted_labels):
    num_samples = true_labels.shape[0]
    result = np.zeros(num_samples, dtype=np.float64)
    # pylint: disable=not-an-iterable
    for i in prange(num_samples):
        counts = {}
        for label in predicted_labels[i]:
            if label in counts:
                counts[label] += 1
            else:
                counts[label] = 1

        # find labels with max count
        max_count = -1
        majority_votes = []
        for label, count in counts.items():
            if count > max_count:
                max_count = count
                majority_votes = [label]
            elif count == max_count:
                majority_votes.append(label)

        # Check if true label is in majority votes
        if true_labels[i] in majority_votes:
            result[i] = 1.0 / len(majority_votes)
    return result


def get_nbr_idxs(
    data,
    n_nbrs=15,
    method="exact",
    metric="euclidean",
    method_kwds=None,
    nbrs=None,
    verbose=False,
):
    n_calc_nbrs = n_nbrs + 1
    calc_nbrs = True
    if nbrs is not None:
        calc_nbrs = n_calc_nbrs > nbrs.idx.shape[1]

    if calc_nbrs:
        nbrs = calculate_neighbors(
            data=data,
            n_neighbors=n_calc_nbrs,
            metric=metric,
            method=method,
            return_distance=False,
            method_kwds=method_kwds,
            verbose=verbose,
        )
    if nbrs is None:
        raise ValueError("nbrs is None")

    return nbrs.idx[:, 1:n_calc_nbrs]


def label_encode(arr):
    # Get the unique values in the array
    unique_values = np.unique(arr)

    # Create a dictionary mapping each unique value to an integer
    value_to_int = {value: i for i, value in enumerate(unique_values)}

    # Use the dictionary to map each string value to its corresponding integer
    return np.array([value_to_int[value] for value in arr])


def get_labels(target, label_id=-1):
    if isinstance(target, np.ndarray):
        if not isinstance(label_id, int):
            raise ValueError("label_id must be an integer when target is an array")
        return label_encode(target[:, label_id])
    if isinstance(target, pd.DataFrame):
        if isinstance(label_id, int):
            return label_encode(target.iloc[:, label_id].values)
        return label_encode(target.loc[:, label_id].values)
    if isinstance(target, range):
        return None
    raise ValueError(f"Unknown target type {type(target)}")


@dataclass
class LabelPreservationEval(EmbeddingEval):
    metric: str = "euclidean"
    n_neighbors: Union[List[int], int] = 15  # can also be a list
    verbose: bool = False
    label_id: Union[int, str] = -1

    def listify_n_neighbors(self):
        if not isinstance(self.n_neighbors, List):
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

        if ctx is None:
            raise ValueError("ctx is None")

        try:
            target = read_target(
                ctx.dataset_name,
                drnb_home=ctx.drnb_home,
            )
        except FileNotFoundError:
            return None
        labels = get_labels(target, self.label_id)
        return labels

    def evaluatev(self, _, coords, ctx=None):
        labels = self._evaluate_setup(ctx)

        return label_presv(
            coords,
            labels,
            n_nbrs=self.n_neighbors,
            method="exact",
            metric=self.metric,
            method_kwds=None,
            nbrs=None,
        )

    def evaluate(self, X, coords, ctx=None):
        labels = self._evaluate_setup(ctx)

        lps = label_pres(
            coords,
            labels,
            n_nbrs=self.n_neighbors,
            method="exact",
            metric=self.metric,
            method_kwds=None,
            nbrs=None,
        )
        return [
            EvalResult(
                eval_type="LP",
                label=self.to_str(n_nbrs),
                info=dict(metric=self.metric, n_neighbors=n_nbrs),
                value=lp,
            )
            for n_nbrs, lp in zip(self.n_neighbors, lps)
        ]

    def to_str(self, n_neighbors):
        return f"lp-{n_neighbors}-{self.metric}"

    def __str__(self):
        return self.to_str(self.n_neighbors)
