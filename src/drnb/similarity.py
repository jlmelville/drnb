import numpy as np
import umap


def smooth_knn(
    knn_indices, knn_dists, n_neighbors, local_connectivity=1, bandwidth=1.0
):
    knn_dists = knn_dists.astype(np.float32)

    sigmas, rhos = umap.umap_.smooth_knn_dist(
        knn_dists,
        float(n_neighbors),
        local_connectivity=float(local_connectivity),
        bandwidth=bandwidth,
    )

    rows, cols, vals, dists = umap.umap_.compute_membership_strengths(
        knn_indices, knn_dists, sigmas, rhos, return_dists=True
    )
    return rows, cols, vals, dists
