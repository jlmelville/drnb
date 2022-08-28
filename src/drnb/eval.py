import numpy as np


# https://github.com/YingfanWang/PaCMAP/blob/c7c45dbd0fec7736764d0e28203eb0e0515f3427/evaluation/evaluation.py
def random_triplet_eval(
    X, X_new, triplets=None, random_state=None, n_triplets_per_point=5
):
    n_obs = X.shape[0]

    # Sampling Triplets
    # Five triplet per point
    if triplets is None:
        triplets = get_triplets(
            X, seed=random_state, n_triplets_per_point=n_triplets_per_point
        )
    else:
        if (
            len(triplets.shape) != 3
            or triplets.shape[2] != 2
            or triplets.shape[0] != n_obs
        ):
            raise ValueError(
                f"triplets should have shape ({n_obs}, {n_triplets_per_point}, 2)"
            )
        n_triplets_per_point = triplets.shape[1]

    # 3D (Nx1x1) array where e.g. anchors[i][0][0] = i: [[[0]], [[1]], [[2]] ... [[n_obs]]]
    anchors = np.arange(n_obs).reshape((-1, 1, 1))

    # Calculate the distances and generate labels
    # broadcasting to b flattens the anchors and triplets to generate a list of pairs:
    # [0, triplet00], [0, triplet01], [0, triplet10], [0, triplet11] etc.
    b = np.broadcast(anchors, triplets)
    labels = calc_labels(X, b)

    # Calculate distances for LD
    b = np.broadcast(anchors, triplets)
    pred_vals = calc_labels(X_new, b)

    correct = np.sum(pred_vals == labels)
    acc = correct / n_obs / n_triplets_per_point
    return acc


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
