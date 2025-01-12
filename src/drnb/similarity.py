from typing import Tuple

import numpy as np
import umap


def smooth_knn(
    knn_indices: np.ndarray,
    knn_dists: np.ndarray,
    n_neighbors: int,
    local_connectivity: float = 1.0,
    bandwidth: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the unsymmetrized k-nearest neighbors affinity graph from UMAP. Data is
    returned as a tuple of (rows, cols, vals, dists) where rows and cols are the row and
    column indices of the affinity matrix, vals are the affinity values, and dists are the
    distances between the points."""
    knn_dists = knn_dists.astype(np.float32)

    sigmas, rhos = umap.umap_.smooth_knn_dist(
        knn_dists,
        k=float(n_neighbors),
        local_connectivity=float(local_connectivity),
        bandwidth=bandwidth,
    )

    rows, cols, vals, dists = umap.umap_.compute_membership_strengths(
        knn_indices, knn_dists, sigmas, rhos, return_dists=True
    )
    return rows, cols, vals, dists
