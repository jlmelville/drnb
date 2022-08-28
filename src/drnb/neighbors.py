import sklearn.metrics


def dmat(x):
    if isinstance(x, tuple) and len(x) == 2:
        x = x[0]
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()
    return sklearn.metrics.pairwise_distances(x)
