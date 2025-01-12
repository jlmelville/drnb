from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numba import njit, prange
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from drnb.log import is_progress_report_iter, log


def svd_shuffle(
    data: np.ndarray | csr_matrix,
    max_col: int = 1000,
    n_shuffles: int = 10,
    algorithm: Literal["arpack", "randomized"] = "arpack",
):
    """Perform a Singular Value Decomposition (SVD) on the data and compare the actual
    explained variance to shuffled data. This can be used to determine the number of
    components to keep in the SVD.
    max_col: int
        The maximum number of components to extract. Extract too few and you may not
        reach the point where the real variance is less than the shuffled variance.
        Extracting too many is computationally expensive.
    n_shuffles: int
        The number of shuffles to perform
    algorithm: str
        The algorithm to use for the SVD. Either 'arpack' or 'randomized'
    """
    log.info("commencing calibration")
    max_col = np.min([max_col, data.shape[0] - 1, data.shape[1] - 1])

    svd = TruncatedSVD(n_components=max_col, algorithm=algorithm)
    _ = svd.fit_transform(data)
    real_variances = svd.explained_variance_ratio_
    log.info("initial SVD complete")

    permuted_variances = np.zeros((n_shuffles, max_col))

    for shuffle in range(n_shuffles):
        data_shuffled = shuffle_data(data)
        _ = svd.fit_transform(data_shuffled)
        permuted_variances[shuffle] = svd.explained_variance_ratio_
        if is_progress_report_iter(shuffle, n_shuffles, max_progress_updates=10):
            log.info("permutation run %d completed", shuffle + 1)

    pvals = []
    for i in range(max_col):
        permuted_sorted = np.sort(permuted_variances[:, i])
        p_value = (np.searchsorted(permuted_sorted, real_variances[i]) + 1) / (
            n_shuffles + 1
        )
        pvals.append(p_value)

    df = pd.DataFrame(
        {
            "n_components": range(1, max_col + 1),
            "actual": np.cumsum(real_variances),
            "shuffled": np.mean(np.cumsum(permuted_variances, axis=1), axis=0),
        }
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=pd.melt(df, ["n_components"], var_name="data"),
        x="n_components",
        y="value",
        hue="data",
    )

    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Variance")
    plt.title("Var Plot")

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(1, max_col + 1), y=pvals)
    sns.lineplot(x=range(1, max_col + 1), y=0.05)


def shuffle_data(data: csr_matrix | np.ndarray) -> csr_matrix | np.ndarray:
    """Shuffle the columns of a CSR matrix or a 2D numpy array"""
    if isinstance(data, csr_matrix):
        return shuffle_csr_csc_parallel(data)
    if isinstance(data, np.ndarray):
        return shuffle_array(data)
    raise ValueError(f"Don't know how to shuffle data of type {data.__class__}")


def shuffle_csr_csc_parallel(data: csr_matrix) -> csr_matrix:
    """Shuffle the columns of a CSR matrix in parallel"""
    csc = data.tocsc()
    permute_csc_data_indices(csc.data, csc.indices, csc.indptr, csc.shape[0])
    return csc.tocsr()


@njit(parallel=True)
def permute_csc_data_indices(
    data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, nrows: int
):
    """Permute the data and indices of a CSC matrix in parallel"""
    # pylint:disable=not-an-iterable
    for i in prange(indptr.shape[0] - 1):
        start_idx = indptr[i]
        end_idx = indptr[i + 1]
        column_data = data[start_idx:end_idx]
        column_indices = indices[start_idx:end_idx]

        # Generate a permutation of all possible indices and use the first
        # len(column_indices) as the new indices
        new_indices = np.random.permutation(nrows)[: len(column_indices)]

        sort_mask = np.argsort(new_indices)
        data[start_idx:end_idx] = column_data[sort_mask]
        indices[start_idx:end_idx] = new_indices[sort_mask]


def shuffle_array(data: np.ndarray) -> np.ndarray:
    """Shuffle the columns of a 2D numpy array"""
    data_shuffled = np.empty(data.shape)
    for i in range(data.shape[1]):
        data_shuffled[:, i] = np.random.permutation(data[:, i])
    return data_shuffled
