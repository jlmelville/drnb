"""
This is effectively vendored from the `drnb` package. There are no extra dependencies
because PaCMAP already uses numba.

This module provides a reference implementation of the "locally scaled neighbors"
procedure, based on the method described by Zelnik-Manor and Perona (2004) for
self-tuning spectral clustering. It borrows ideas from TriMap and PaCMAP, where
local scaling factors are used to either weight or select neighbors in
dimensionality reduction.

Usage Example:
    import numpy as np
    from locally_scaled_knn import locally_scaled_neighbors

    # Suppose we have:
    #   idx:   (N, k) array of neighbor indices
    #   dist:  (N, k) array of neighbor distances
    #   l = 15    # number of locally scaled neighbors we want
    #   m = 50    # number of extended neighbors to sample from
    #
    # The first column of idx is assumed to be the self-neighbor with distance=0.
    # We then compute the top-l neighbors after locally scaling:
    scaled_idx, scaled_dist = locally_scaled_neighbors(
        idx, dist, l=15, m=50, scale_from=5, scale_to=7
    )
    # scaled_idx and scaled_dist will each be of shape (N, l).
    # They contain the selected neighbors and original distances, respectively,
    # after local scaling was used to reorder them.
"""

from typing import Tuple

import numba
import numpy as np


@numba.njit(parallel=True)
def _compute_sigma(
    distances: np.ndarray, scale_from: int = 4, scale_to: int = 6
) -> np.ndarray:
    """
    Compute the local scaling factor σᵢ for each point i by taking the average of
    distances to neighbors in the range [scale_from, scale_to], inclusive.

    The default values for scale_from and scale_to are 4 and 6, which correspond
    to the fifth and seventh neighbors, respectively. This is based on how local
    scaling is used in PaCMAP and TriMap and assumes the first column is the
    self-neighbor. You may need to adjust these values (e.g. subtract 1) if
    the neighbor distances do not include the self-neighbor.

    Parameters
    ----------
    distances : np.ndarray
        (N, k) array of distances to the k-nearest neighbors. This can include
        the self-neighbor in the first column (but this is not required).
    scale_from : int, optional
        The index of the first neighbor used to compute the local scale.
        Default is 4 (fifth neighbor, zero-based), assuming the first column
        is the self-neighbor.
    scale_to : int, optional
        The index of the last neighbor used to compute the local scale.
        Default is 6 (seventh neighbor, zero-based), assuming the first column
        is the self-neighbor.

    Returns
    -------
    sigma : np.ndarray
        (N,) array of local scaling values for each point i. Values are clamped to
        be at least 1e-10 to avoid division by zero.
    """
    n = distances.shape[0]
    sigma = np.empty(n, dtype=distances.dtype)

    start_col = scale_from
    end_col = scale_to + 1  # end_col is exclusive in Python

    for i in numba.prange(n):
        dist_slice = distances[i, start_col:end_col]
        if dist_slice.size > 0:
            mean_val = 0.0
            for val in dist_slice:
                mean_val += val
            mean_val /= dist_slice.size
            sigma[i] = max(mean_val, 1e-10)  # clamp to avoid divide-by-zero
        else:
            sigma[i] = 1e-10

    return sigma


@numba.njit(parallel=True)
def _compute_locally_scaled_distances(
    indices: np.ndarray, distances: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """
    Compute the locally scaled distances:
        LSD(i, j) = (distance(i, j)^2) / (sigma[i] * sigma[j])
    where sigma[i] is the local scaling factor for point i, and sigma[j]
    for point j. The neighbor j for point i is found via indices[i, j].

    Parameters
    ----------
    indices : np.ndarray
        (N, k) array of neighbor indices for each point i.
    distances : np.ndarray
        (N, k) array of distances corresponding to the neighbor indices.
    sigma : np.ndarray
        (N,) array containing the local scaling factor for each point.

    Returns
    -------
    lsd : np.ndarray
        (N, k) array of locally scaled distances LSD(i, j).
    """
    n, k = distances.shape
    lsd = np.empty_like(distances)

    for i in numba.prange(n):
        sigma_i = sigma[i]
        for nbr_idx in range(k):
            j = indices[i, nbr_idx]
            dist_ij = distances[i, nbr_idx]
            val = dist_ij * dist_ij / (sigma_i * sigma[j])
            lsd[i, nbr_idx] = val

    return lsd


def locally_scaled_neighbors(
    indices: np.ndarray,
    distances: np.ndarray,
    l: int,
    m: int | None = None,
    scale_from: int = 4,
    scale_to: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select the l-nearest locally-scaled neighbors for each point out of the
    top-m neighbors given precomputed k-nearest neighbors (where k >= m).

    The procedure is:
      1) Compute local scaling factors sigma[i] for each point i, using the
         average distance of neighbors in [scale_from..scale_to] (inclusive).
      2) For each row i, consider only the columns from 0..(m-1) in the
         neighbor list. Compute LSD(i, j) = (dist_ij^2) / (sigma[i]*sigma[j]).
      3) Reorder those m neighbors in ascending order by LSD.
      4) Select the top-l neighbors under LSD.
      5) Finally, reorder those l neighbors in ascending order by their
         original (unscaled) distance, and return them.

    Parameters
    ----------
    indices : np.ndarray
        (N, k) array containing the k-nearest neighbors for each point i.
        indices[i, 0] is assumed to be i itself (the self neighbor).
    distances : np.ndarray
        (N, k) array containing distances corresponding to indices.
        distances[i, 0] is assumed to be 0.0 (self distance).
    l : int
        Number of neighbors to retain after local scaling.
    m : int | None, optional
        Number of extended neighbors to consider for local scaling. Must satisfy
        m <= k. If None, uses all available neighbors (m = k). Default is None.
    scale_from : int, optional
        Index of the first neighbor used to compute the local scale.
        Default is 4 (fifth neighbor, zero-based).
    scale_to : int, optional
        Index of the last neighbor used to compute the local scale.
        Default is 6 (seventh neighbor, zero-based).

    Returns
    -------
    final_indices : np.ndarray
        (N, l) array of the l selected neighbor indices for each point.
    final_distances : np.ndarray
        (N, l) array of the corresponding original distances for each point.
        Reordered in ascending order of the original distance.

    Raises
    ------
    ValueError
        If m is greater than k or l is greater than m.

    Notes
    -----
    The default values for scale_from and scale_to are 4 and 6, which correspond
    to the fifth and seventh neighbors, respectively. This is based on how local
    scaling is used in PaCMAP and TriMap and assumes the first column is the
    self-neighbor. You may need to adjust these values (e.g. subtract 1) if
    the neighbor distances do not include the self-neighbor.
    """
    n, k = distances.shape
    if m is None:
        m = k
    if k < m:
        raise ValueError(f"m must be <= k. Found {k=}, {m=}.")
    if l > m:
        raise ValueError(f"l must be <= m. Found {l=}, {m=}.")

    # 1) Compute the local scaling factor sigma[i] for each point.
    sigma = _compute_sigma(distances, scale_from=scale_from, scale_to=scale_to)

    # 2) For each row, compute LSD for the first m neighbors.
    truncated_indices = indices[:, :m]
    truncated_distances = distances[:, :m]

    lsd = _compute_locally_scaled_distances(
        truncated_indices, truncated_distances, sigma
    )

    # 3) Reorder those m neighbors by LSD ascending, 4) select top-l.
    #    We'll use np.argsort along axis=1 for simplicity.
    #    Then we reorder the original distances and indices accordingly.
    sorted_lsd_idx = np.argsort(lsd, axis=1)  # shape (N, m)

    # We'll build arrays to hold the top-l by LSD.
    l_indices = np.empty((n, l), dtype=indices.dtype)
    l_distances = np.empty((n, l), dtype=distances.dtype)

    # For each row i, gather top-l neighbors by LSD.
    for i in range(n):
        # top-l columns under LSD
        best_cols = sorted_lsd_idx[i, :l]
        l_indices[i, :] = truncated_indices[i, best_cols]
        l_distances[i, :] = truncated_distances[i, best_cols]

    # 5) Reorder those l neighbors by the original distance.
    final_indices = np.empty_like(l_indices)
    final_distances = np.empty_like(l_distances)

    # We'll do another argsort, but now on l_distances.
    for i in range(n):
        order_l = np.argsort(l_distances[i, :])
        final_indices[i, :] = l_indices[i, order_l]
        final_distances[i, :] = l_distances[i, order_l]

    return final_indices, final_distances
