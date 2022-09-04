import numpy as np


def get_triplets(X, seed=None, n_triplets_per_point=5):
    anchors = np.arange(X.shape[0])
    rng = np.random.default_rng(seed=seed)
    # for each row of X generate n_triplets_per_point pairs sampled from anchors
    triplets = rng.choice(anchors, (X.shape[0], n_triplets_per_point, 2))
    return triplets


def calc_distances(X, pairs):
    distances = np.empty(pairs.shape)
    distances.flat = [np.linalg.norm(X[u] - X[v]) for (u, v) in pairs]
    return distances


def calc_labels(X, pairs):
    distances = calc_distances(X, pairs)
    return distances[:, :, 0] < distances[:, :, 1]


def validate_triplets(triplets, n_obs):
    if len(triplets.shape) != 3 or triplets.shape[2] != 2 or triplets.shape[0] != n_obs:
        raise ValueError(
            f"triplets should have shape ({n_obs}, n_triplets_per_point, 2)"
        )
