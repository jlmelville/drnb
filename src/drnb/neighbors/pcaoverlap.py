"""A module for computing nearest neighbor overlap between PCA results and raw data,
an alternative to looking at variance explained."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from drnb.eval.nbrpres import nn_acc
from drnb.io.dataset import read_data
from drnb.log import log
from drnb.neighbors.compute import calculate_exact_neighbors
from drnb.neighbors.store import read_neighbors

DEFAULT_COMPONENTS = [1, 2, 5, 10, 20, 50, 100, 150]


def pca_nn_overlap(
    dataset_name: str,
    n_neighbors: int = 150,
    n_neighbors_small: int | None = 15,
    components: list[int] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Compute nearest neighbor overlap between PCA results and raw data. Plots the
    results as a line graph of overlap against the number of PCA components and returns
    a DataFrame with the results. Assumes neighbors include self (include_self=True).

    Args:
        dataset_name: name of the dataset
        n_neighbors: number of neighbors to compare overlap with.
        n_neighbors_small (optional): A smaller number of neighbors to compare overlap
        with. Used to answer the specific question: "are the top PCA neighbors in the
        extended neighborhood of the original data?"
        components (optional): list of number of PCA components to compute overlap for.
        If not provided, a default list of components between 1 and 150 will be used.
        verbose: whether to print verbose output

    Returns:
        A DataFrame with columns:
          - n_components: number of PCA components
          - overlap_big_big: overlap between PCA and raw data for k=n_neighbors
          - overlap_small_small (optional): overlap between PCA and raw data for
            k=n_neighbors_small if provided.
          - overlap_big_small (optional): overlap between raw data with k=n_neighbors
            and PCA with k=n_neighbors_small, i.e. does the original data contain the
            top PCA neighbors in an extended neighborhood? Only returned if
            n_neighbors_small is provided.
    """
    if components is None:
        components = DEFAULT_COMPONENTS

    data = read_data(
        dataset=dataset_name,
        verbose=verbose,
    )
    n_items, n_features = data.shape

    n_neighbors, n_neighbors_small, components, max_components = _validate(
        n_neighbors,
        n_neighbors_small,
        components,
        n_items,
        n_features,
    )

    # get exact neighbors in the original space
    raw_neighbors = read_neighbors(
        name=dataset_name,
        n_neighbors=n_neighbors,
        metric="euclidean",
        exact=True,
        return_distance=False,
        verbose=verbose,
    )
    if raw_neighbors is None:
        raise ValueError(
            f"No cached neighbors found for dataset '{dataset_name}' "
            f"(n_neighbors={n_neighbors}, metric='euclidean', exact=True)."
        )

    # PCA to the max components to get all needed scores
    pca = PCA(n_components=max_components)
    pca_scores = pca.fit_transform(data)

    curve = create_overlap_curve(
        raw_neighbors.idx,
        pca_scores,
        n_neighbors,
        n_neighbors_small,
        components,
        verbose,
    )

    plot_overlap_curve(
        curve,
        n_neighbors,
        n_neighbors_small,
        dataset_name=dataset_name,
    )

    return curve


def create_overlap_curve(
    raw_idx: np.ndarray,
    pca_scores: np.ndarray,
    n_neighbors: int | None = None,
    n_neighbors_small: int | None = None,
    components: list[int] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """compute nearest neighbor overlap between PCA results and raw data.
    Results are returned in a DataFrame with columns:
      - n_components: number of PCA components
      - overlap_big_big: overlap between PCA and raw data for k=n_neighbors
      - overlap_small_small (optional): overlap between PCA and raw data for
        k=n_neighbors_small if provided.
      - overlap_big_small (optional): overlap between raw data with k=n_neighbors and
        PCA with k=n_neighbors_small, i.e. does the original data contain the top PCA
        neighbors in an extended neighborhood? Only returned if n_neighbors_small is
        provided. Assumes neighbor indices include self (include_self=True).
    """

    n_items, n_features = pca_scores.shape
    # don't need max_components because we already did the PCA
    n_neighbors, n_neighbors_small, components, _ = _validate(
        n_neighbors,
        n_neighbors_small,
        components,
        n_items,
        n_features,
    )

    # compute overlap for each component
    overlaps_big_big = []
    overlaps_small_small = []
    overlaps_big_small = []
    for n_components in components:
        coords = pca_scores[:, :n_components]
        pca_neighbors = calculate_exact_neighbors(
            data=coords,
            n_neighbors=n_neighbors,
            metric="euclidean",
            return_distance=False,
            include_self=True,
            verbose=verbose,
            quiet_plugin_logs=True,
        )
        pca_idx = pca_neighbors.idx
        overlaps_big_big.append(
            nn_acc(pca_idx[:, :n_neighbors], raw_idx[:, :n_neighbors])
        )
        if n_neighbors_small is not None:
            overlaps_small_small.append(
                nn_acc(pca_idx[:, :n_neighbors_small], raw_idx[:, :n_neighbors_small])
            )
            overlaps_big_small.append(
                nn_acc(raw_idx[:, :n_neighbors], pca_idx[:, :n_neighbors_small])
            )

    curve = pd.DataFrame(
        {
            "n_components": components,
            "overlap_big_big": overlaps_big_big,
        }
    )

    if n_neighbors_small is not None:
        curve["overlap_small_small"] = overlaps_small_small
        curve["overlap_big_small"] = overlaps_big_small

    return curve


def plot_overlap_curve(
    curve: pd.DataFrame,
    n_neighbors: int,
    n_neighbors_small: int | None,
    dataset_name: str | None = None,
) -> None:
    ax = sns.lineplot(
        data=curve,
        x="n_components",
        y="overlap_big_big",
        marker="o",
        label=f"k={n_neighbors}",
    )

    if "overlap_small_small" in curve.columns:
        sns.lineplot(
            data=curve,
            x="n_components",
            y="overlap_small_small",
            marker="o",
            ax=ax,
            label=f"k={n_neighbors_small}",
        )
    if "overlap_big_small" in curve.columns:
        sns.lineplot(
            data=curve,
            x="n_components",
            y="overlap_big_small",
            marker="o",
            ax=ax,
            label=f"raw k={n_neighbors} vs PCA k={n_neighbors_small}",
        )

    plt.xlabel("PCA components")
    plt.ylabel(f"Neighbor overlap")
    plt.legend()
    if dataset_name is not None:
        title_prefix = f"{dataset_name}: "
    else:
        title_prefix = ""
    plt.title(f"{title_prefix}neighbor overlap vs PCA components")
    plt.tight_layout()
    plt.show()


def _validate(
    n_neighbors: int,
    n_neighbors_small: int | None,
    components: list[int] | None,
    n_items: int,
    n_features: int,
) -> tuple[int, int | None, list[int], int]:
    # validation on number of neighbors, components

    if n_neighbors > n_items:
        log.warning(
            "Requested %d neighbors but only %d are available",
            n_neighbors,
            n_items,
        )
        n_neighbors = n_items

    if components is None:
        components = DEFAULT_COMPONENTS
    components = sorted(components)
    max_components = components[-1]

    max_rank = min(n_items, n_features)

    if max_components > max_rank:
        log.warning(
            "Requested maximum %d components but data has max rank %d",
            max_components,
            max_rank,
        )
        max_components = max_rank

    if n_neighbors_small is not None:
        if n_neighbors_small > n_items or n_neighbors_small == n_neighbors:
            log.warning(
                "Requested %d invalid 'small' neighbors " + " setting to None",
                n_neighbors_small,
            )
            n_neighbors_small = None

    return n_neighbors, n_neighbors_small, components, max_components
