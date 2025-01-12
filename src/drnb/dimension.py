import numpy as np


# Levina and Bickel 2004
# https://proceedings.neurips.cc/paper/2004/hash/74934548253bcab8490ebd74afed7031-Abstract.html
def mle_local(
    knn_dist: np.ndarray,
    eps: float = 1.0e-10,
    n_neighbors: int | None = None,
    remove_self: bool = False,
) -> np.ndarray:
    """Compute local intrinsic dimensionality using the maximum likelihood estimation
    method proposed by Levina and Bickel (2004)."""
    if remove_self:
        # remove self-neighbor
        knn_dist = knn_dist[:, 1:]
    knn_dist[knn_dist < eps] = eps
    log_knn = np.log(knn_dist)
    if n_neighbors is None:
        n_neighbors = knn_dist.shape[1]
    elif n_neighbors > knn_dist.shape[1]:
        raise ValueError(
            f"n_neighbors must be <= {knn_dist.shape[1]} but was {n_neighbors}"
        )

    k1 = n_neighbors - 1
    log_rij = -np.sum(log_knn[:, :-1], axis=1)
    return k1 / (k1 * log_knn[:, -1] + log_rij)


# MacKay and Ghahramani 2005
# http://www.inference.org.uk/mackay/dimension/
def mle_global(
    knn_dist: np.ndarray,
    eps: float = 1.0e-10,
    n_neighbors: int | None = None,
    remove_self: bool = False,
) -> float:
    """Compute global intrinsic dimensionality using the maximum likelihood estimation
    method proposed by MacKay and Ghahramani (2005)."""
    id_local = mle_local(
        knn_dist, eps=eps, n_neighbors=n_neighbors, remove_self=remove_self
    )
    return 1.0 / np.mean(1.0 / id_local)
