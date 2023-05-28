from collections import Counter
from dataclasses import dataclass, field
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
    labels,
    nbr_idxs,
    n_neighbors: List[int],
    balanced,
):
    if labels is None:
        log.warning("No labels provided")
        return np.full((len(n_neighbors),), np.nan)

    counts = get_counts(labels)
    max_n_nbrs = nbr_idxs.shape[1]
    lps = []
    num_classes = np.max(labels) + 1
    for nbrs in n_neighbors:
        if nbrs <= max_n_nbrs:
            mv = _label_pres_impl(labels, nbr_idxs, nbrs, counts)
            if balanced:
                mean = np.sum(mv / (counts[labels] * num_classes))
            else:
                mean = np.mean(mv)
            lps.append(mean)
        else:
            lps.append(np.nan)
    return np.array(lps)


def label_presv(
    labels,
    nbr_idxs,
    n_neighbors: List[int],
):
    if labels is None:
        log.warning("No labels provided")
        return np.full((len(n_neighbors), len(labels)), np.nan)

    counts = get_counts(labels)
    return np.array(
        [_label_pres_impl(labels, nbr_idxs, nbrs, counts) for nbrs in n_neighbors]
    )


def _label_pres_impl(labels, nbr_idxs, nbrs, counts=None):
    if nbrs > nbr_idxs.shape[1]:
        return [np.nan] * len(labels)
    predicted_labels = labels[nbr_idxs[:, :nbrs]]
    return majority_vote(labels, predicted_labels, counts)


def majority_vote(true_labels, predicted_labels, counts=None):
    if counts is None:
        counts = get_counts(true_labels)
    return majority_vote_impl(true_labels, predicted_labels, counts)


@njit(parallel=True)
def majority_vote_impl(true_labels, predicted_labels, counts):
    num_samples = true_labels.shape[0]
    result = np.zeros(num_samples, dtype=np.float64)
    # pylint: disable=not-an-iterable
    for i in prange(num_samples):
        votes = {}
        for label in predicted_labels[i]:
            if label in votes:
                votes[label] += 1
            else:
                votes[label] = 1

        # find labels with max vote
        max_vote = -1
        majority_votes = []
        for label, vote in votes.items():
            if vote > max_vote:
                max_vote = vote
                majority_votes = [label]
            elif vote == max_vote:
                majority_votes.append(label)

        # Check if true label is in majority votes
        if true_labels[i] in majority_votes:
            majority_votes = np.array(majority_votes)
            # for ties, use the frequency with which we would guess the correct
            # label based on the relative frequencies of the tied labels in the
            # dataset as a whole
            result[i] = counts[true_labels[i]] / np.sum(counts[majority_votes])
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


def get_counts(labels):
    counts = count_integers(labels)
    return np.array([counts[label] for label in labels])


def count_integers(lst: np.ndarray) -> np.ndarray:
    if len(lst) == 0:
        return np.array([])
    counter = Counter(lst)
    max_value = np.max(lst)
    counts = [counter[i] for i in range(max_value + 1)]
    return np.array(counts)


def label_encode(arr):
    # Get the unique values in the array
    unique_values = np.unique(arr)

    # Create a dictionary mapping each unique value to an integer
    value_to_int = {value: i for i, value in enumerate(unique_values)}

    # Use the dictionary to map each string value to its corresponding integer
    return np.array([value_to_int[value] for value in arr])


def get_labels(target, label_id: Union[int, str] = -1):
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
    n_neighbors: List[int] = field(default_factory=lambda: [15])
    label_id: Union[int, str] = -1
    balanced: bool = True
    verbose: bool = False

    def requires(self):
        return dict(
            name="neighbors",
            metric=self.metric,
            n_neighbors=int(np.max(self.n_neighbors)),
        )

    def _evaluate_setup(self, coords, ctx):
        if ctx is None:
            raise ValueError("ctx is None")

        try:
            target = read_target(
                ctx.dataset_name,
                drnb_home=ctx.drnb_home,
            )
        except FileNotFoundError as exc:
            raise ValueError("Can't find target file") from exc

        labels = get_labels(target, self.label_id)

        max_n_nbrs = max(self.n_neighbors)
        nbr_idxs = get_nbr_idxs(
            data=coords,
            n_nbrs=max_n_nbrs,
            method="exact",
            metric="euclidean",
            method_kwds=None,
            nbrs=None,
            verbose=False,
        )

        return labels, nbr_idxs

    def evaluatev(self, _, coords, ctx=None):
        labels, nbr_idxs = self._evaluate_setup(coords, ctx)

        return label_presv(
            labels,
            nbr_idxs,
            self.n_neighbors,
        )

    def evaluate(self, _, coords, ctx=None):
        labels, nbr_idxs = self._evaluate_setup(coords, ctx)

        lps = label_pres(
            labels,
            nbr_idxs,
            self.n_neighbors,
            self.balanced,
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
        if self.balanced:
            balance_indicator = "b"
        else:
            balance_indicator = ""
        if self.metric == "euclidean":
            metric_indicator = ""
        else:
            metric_indicator = f"-{self.metric}"
        return f"lp{balance_indicator}-{n_neighbors}{metric_indicator}"

    def __str__(self):
        return self.to_str(self.n_neighbors)
