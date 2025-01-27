import numpy as np


def _preprocess_knn_dist(
    knn_dist: np.ndarray,
    eps: float = 1.0e-10,
    n_neighbors: int | None = None,
    remove_self: bool = False,
) -> tuple[np.ndarray, int]:
    """Preprocess k-nearest neighbor distances. Returns log distances and number of
    neighbors used (if the latter was not provided)."""
    if remove_self:
        knn_dist = knn_dist[:, 1:]

    knn_dist[knn_dist < eps] = eps

    if n_neighbors is None:
        n_neighbors = knn_dist.shape[1]
    elif n_neighbors > knn_dist.shape[1]:
        raise ValueError(
            f"n_neighbors must be <= {knn_dist.shape[1]} but was {n_neighbors}"
        )

    # Use only requested number of neighbors
    knn_dist = knn_dist[:, :n_neighbors]
    log_knn = np.log(knn_dist)

    return log_knn, n_neighbors


def _compute_denominators(log_knn: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Compute denominators used in both local and global MLE calculations.

    Parameters
    ----------
    log_knn : np.ndarray
        Log of k-nearest neighbor distances
    n_neighbors : int
        Number of neighbors used

    Returns
    -------
    np.ndarray
        Array of denominators: (k-1)log(r_k) - sum(log(r_j))
    """

    # This doesn't exactly resemble the formula in the Levina and Bickel paper, but
    # it's equivalent, I just did some algebra to simplify the expression.
    # Estimate is 1/mk, where mk is the mean of the log ratio of the furthest
    # neighbor distance to the other neighbors.
    # mk = ∑(log(Rk / Rj)) / (k - 1) (summing for j=1 to k-1)
    # mk = (∑log(Rk) - ∑log(Rj)) / (k - 1)
    # mk = [(k - 1)log(Rk) - ∑log(Rj)] / (k - 1)
    # 1 / mk = (k - 1) / [(k - 1)log(Rk) - ∑log(Rj)]
    # Then it turns out we can also use this for the global estimate too

    k_minus_1 = n_neighbors - 1
    log_rk = log_knn[:, -1]
    sum_log_rij = np.sum(log_knn[:, :-1], axis=1)
    return k_minus_1 * log_rk - sum_log_rij


# Levina and Bickel 2004
# https://proceedings.neurips.cc/paper/2004/hash/74934548253bcab8490ebd74afed7031-Abstract.html
def mle_local(
    knn_dist: np.ndarray,
    eps: float = 1.0e-10,
    n_neighbors: int | None = None,
    remove_self: bool = False,
) -> np.ndarray:
    """Compute local intrinsic dimensionality using the maximum likelihood estimation
    method proposed by Levina and Bickel (2004).

    Parameters
    ----------
    knn_dist : np.ndarray
        Distances to k nearest neighbors, shape (n_samples, k)
    eps : float, default=1.0e-10
        Small constant to avoid log(0)
    n_neighbors : int, optional
        Number of neighbors to use. If None, uses all available neighbors
    remove_self : bool, default=False
        If True, removes the first neighbor (assumed to be self)

    Returns
    -------
    np.ndarray
        Local intrinsic dimension estimate for each point
    """
    log_knn, n_neighbors = _preprocess_knn_dist(knn_dist, eps, n_neighbors, remove_self)
    denominators = _compute_denominators(log_knn, n_neighbors)
    return (n_neighbors - 1) / denominators


# MacKay and Ghahramani 2005
# http://www.inference.org.uk/mackay/dimension/
def mle_global(
    knn_dist: np.ndarray,
    eps: float = 1.0e-10,
    n_neighbors: int | None = None,
    remove_self: bool = False,
) -> float:
    """Compute global intrinsic dimensionality using the MacKay-Ghahramani MLE method,
    which is a harmonic mean of the local Levina-Bickel MLE estimates.

    Parameters
    ----------
    knn_dist : np.ndarray
        Distances to k nearest neighbors, shape (n_samples, k)
    eps : float, default=1.0e-10
        Small constant to avoid log(0)
    n_neighbors : int, optional
        Number of neighbors to use. If None, uses all available neighbors
    remove_self : bool, default=False
        If True, removes the first neighbor (assumed to be self)

    Returns
    -------
    float
        Global intrinsic dimension estimate
    """
    # Usual version would be something like: 1.0 / np.mean(1.0 / id_local), but by
    # doing the algebra, it turns out we can be a bit more efficient by computing the
    # harmonic mean of mk directly.
    # mk = N(k - 1) / ∑_i[(k - 1) log(R_ik) - ∑_j log(R_ij)]
    log_knn, n_neighbors = _preprocess_knn_dist(knn_dist, eps, n_neighbors, remove_self)
    denominators = _compute_denominators(log_knn, n_neighbors)

    # Return harmonic mean multiplied by (k-1)
    n_points = knn_dist.shape[0]
    return n_points * (n_neighbors - 1) / np.sum(denominators)
