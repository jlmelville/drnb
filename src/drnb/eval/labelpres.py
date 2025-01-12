from collections import Counter
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from numba import njit, prange

from drnb.embed.context import EmbedContext
from drnb.eval.base import EmbeddingEval, EvalResult
from drnb.io.dataset import read_target
from drnb.log import log
from drnb.neighbors import NearestNeighbors, calculate_neighbors


def label_pres(
    labels: np.ndarray | None,
    nbr_idxs: np.ndarray,
    n_neighbors: List[int],
    balanced: bool = True,
):
    """
    Compute the label preservation for given labels and neighbor indices over multiple
    neighbor values. Returns an array of preservation scores, one for each value in
    n_neighbors.

    If balanced is True, each label is weighted by the inverse of its frequency in the
    dataset, ensuring that less common labels contribute equally to the overall score.
    If balanced is False, the preservation score is calculated as a simple mean without
    any weighting by label frequency.
    """
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
    labels: np.ndarray,
    nbr_idxs: np.ndarray,
    n_neighbors: List[int],
) -> np.ndarray:
    """Compute the label preservation score for each sample in the dataset. The label
    preservation score is the fraction of the nearest neighbors of each sample (in
    `nbr_idxs`) that have the same label as the sample itself (in `labels`).
    The return value is an array of shape (len(n_neighbors), len(labels)) where the ith
    row contains the label preservation score for the ith sample in the dataset at the
    corresponding number of nearest neighbors in `n_neighbors`."""
    if labels is None:
        log.warning("No labels provided")
        return np.full((len(n_neighbors), len(labels)), np.nan)

    counts = get_counts(labels)
    return np.array(
        [_label_pres_impl(labels, nbr_idxs, n_nbrs, counts) for n_nbrs in n_neighbors]
    )


def _label_pres_impl(
    labels: np.ndarray,
    nbr_idxs: np.ndarray,
    n_nbrs: int,
    counts: np.ndarray | None = None,
) -> np.ndarray:
    if n_nbrs > nbr_idxs.shape[1]:
        return [np.nan] * len(labels)
    predicted_labels = labels[nbr_idxs[:, :n_nbrs]]
    return majority_vote(labels, predicted_labels, counts)


def majority_vote(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    counts: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the majority vote score for each sample in the dataset. If the majority
    vote matches the true label, the score is 1.0 and 0.0 if not. In the event of a
    tie between the true label and other labels, the score is the fraction of the
    top-voted part of the dataset that has the same label as the true label."""
    if counts is None:
        counts = get_counts(true_labels)
    return majority_vote_impl(true_labels, predicted_labels, counts)


@njit(parallel=True)
def majority_vote_impl(
    true_labels: np.ndarray, predicted_labels: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    """Compute the majority vote score for each sample in the dataset. If the majority
    vote matches the true label, the score is 1.0 and 0.0 if not. In the event of a
    tie between the true label and other labels, the score is the fraction of the
    top-voted part of the dataset that has the same label as the true label."""
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
        majority_votes = [-1]
        for label, vote in votes.items():
            if vote > max_vote:
                max_vote = vote
                majority_votes = [label]
            elif vote == max_vote:
                majority_votes.append(label)

        # Check if true label is in majority votes
        if true_labels[i] in majority_votes:
            majority_votes = np.array(majority_votes)
            # if only one label and it's the true label then score is 1
            # for ties, use the frequency with which we would guess the correct
            # label based on the relative frequencies of the tied labels in the
            # dataset as a whole
            result[i] = counts[true_labels[i]] / np.sum(counts[majority_votes])
    return result


def get_nbr_idxs(
    data: np.ndarray,
    n_nbrs: int = 15,
    method: str = "exact",
    metric: str = "euclidean",
    method_kwds: dict | None = None,
    nbrs: NearestNeighbors | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Get the indices of the nearest neighbors of each point in the data. If the
    `nbrs` argument is provided, the function will use the existing nearest neighbors
    object to get the indices. Otherwise, it will calculate the nearest neighbors using
    the specified method and metric. The return value is an array where the ith row
    contains the indices of the nearest neighbors of the ith point in the data."""
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


def get_counts(labels: np.ndarray) -> np.ndarray:
    """Get the number of occurrences of each label in a list. The return value is an
    array where the ith element is the number of occurrences over the entire list of
    the label at position i in the input `labels` list. This provides a quick look up
    table for the number of occurrences of each label in the list."""
    counts = count_integers(labels)
    return np.array([counts[label] for label in labels])


def count_integers(lst: np.ndarray) -> np.ndarray:
    """Count the number of occurrences of non-negative integers in a list.
    The return value is an array where the ith element is the number of
    occurrences of the integer i in the input list."""
    if len(lst) == 0:
        return np.array([])
    counter = Counter(lst)
    max_value = np.max(lst)
    counts = [counter[i] for i in range(max_value + 1)]
    return np.array(counts)


def label_encode(arr: np.ndarray) -> np.ndarray:
    """Encode an array of strings as integers. The integers are assigned in the order
    that the unique values appear in the array. Returns a new array with the same shape
    as the input array, but with the strings replaced by integers codes."""
    # Get the unique values in the array
    unique_values = np.unique(arr)

    # Create a dictionary mapping each unique value to an integer
    value_to_int = {value: i for i, value in enumerate(unique_values)}

    # Use the dictionary to map each string value to its corresponding integer
    return np.array([value_to_int[value] for value in arr])


def get_labels(
    target: np.ndarray | pd.DataFrame | range, label_id: int | str = -1
) -> np.ndarray:
    """Get the labels from a target array or DataFrame. If the target is an array, the
    label_id must be an integer indicating the column index of the labels. If the target
    is a DataFrame, the label_id can be an integer or a string indicating the column
    name of the labels. If the target is a range, the labels are assumed to be the
    integers in the range. Returns an array of integer labels, one for each sample in
    the target."""
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
    """Evaluate the preservation of labels in the embedding. The evaluation is based on
    the fraction of nearest neighbors of each sample that have the same label as the
    sample itself. The evaluation is performed for multiple values of n_neighbors and
    the result is the mean preservation score over all samples in the dataset.

    Attributes:
    metric: str - the distance metric to use for nearest neighbor calculations
    n_neighbors: List[int] - the number of nearest neighbors to use in the evaluation
    label_id: int | str - the column index or name of the labels in the target data
    balanced: bool - whether to weight the preservation score by the inverse of the
                        frequency of each label in the dataset
    verbose: bool - whether to print verbose output during the evaluation
    """

    metric: str = "euclidean"
    n_neighbors: List[int] = field(default_factory=lambda: [15])
    label_id: int | str = -1
    balanced: bool = True
    verbose: bool = False

    def requires(self):
        return {
            "name": "neighbors",
            "metric": self.metric,
            "n_neighbors": int(np.max(self.n_neighbors)),
        }

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

    def evaluatev(
        self, _, coords: np.ndarray, ctx: EmbedContext | None = None
    ) -> List[np.ndarray]:
        """Evaluate the per-item label preservation. Return a list of arrays
        of accuracies, one per value of n_neighbors in the evaluation."""
        labels, nbr_idxs = self._evaluate_setup(coords, ctx)

        return label_presv(
            labels,
            nbr_idxs,
            self.n_neighbors,
        )

    def evaluate(
        self, _, coords: np.ndarray, ctx: EmbedContext | None = None
    ) -> EvalResult:
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
                info={"metric": self.metric, "n_neighbors": n_nbrs},
                value=lp,
            )
            for n_nbrs, lp in zip(self.n_neighbors, lps)
        ]

    def to_str(self, n_neighbors: int) -> str:
        """Create a string representation of the evaluation for a given number of
        neighbors."""
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
