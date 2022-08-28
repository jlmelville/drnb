import numpy as np


def get_triplets(X, seed=None):
    # 0..nrow(X)
    anchors = np.arange(X.shape[0])
    rng = np.random.default_rng(seed=seed)
    # for each row of X generate 5 pairs sampled from anchors
    triplets = rng.choice(anchors, (X.shape[0], 5, 2))
    return triplets


# https://github.com/YingfanWang/PaCMAP/blob/c7c45dbd0fec7736764d0e28203eb0e0515f3427/evaluation/evaluation.py
def random_triplet_eval(X, X_new, triplets=None, random_state=42):
    # Sampling Triplets
    # Five triplet per point
    if triplets is None:
        triplets = get_triplets(X, seed=random_state)

    # 3D array where e.g. anchors[i][0][0] = i
    anchors = np.arange(X.shape[0]).reshape((-1, 1, 1))
    # Calculate the distances and generate labels
    b = np.broadcast(anchors, triplets)
    distances = np.empty(b.shape)
    distances.flat = [np.linalg.norm(X[u] - X[v]) for (u, v) in b]

    b = np.broadcast(anchors, triplets)
    labels = distances[:, :, 0] < distances[:, :, 1]

    # Calculate distances for LD
    b = np.broadcast(anchors, triplets)
    distances_l = np.empty(b.shape)
    distances_l.flat = [np.linalg.norm(X_new[u] - X_new[v]) for (u, v) in b]
    pred_vals = distances_l[:, :, 0] < distances_l[:, :, 1]
    correct = np.sum(pred_vals == labels)
    acc = correct / X.shape[0] / 5
    return acc
