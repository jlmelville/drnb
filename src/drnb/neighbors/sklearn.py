from sklearn.neighbors import NearestNeighbors

SKLEARN_METRICS = {
    "cosine": "cosine",
    "manhattan": "manhattan",
    "euclidean": "euclidean",
}

SKLEARN_DEFAULTS = dict(
    algorithm="auto",
    n_jobs=-1,
)


def sklearn_neighbors(
    X,
    n_neighbors=15,
    algorithm="auto",
    metric="euclidean",
    metric_kwds=None,
    n_jobs=-1,
    return_distance=True,
):
    sklearn_space = SKLEARN_METRICS[metric]

    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm=algorithm,
        n_jobs=n_jobs,
        metric=sklearn_space,
        metric_params=metric_kwds,
    ).fit(X)

    knn = nn.kneighbors(X, return_distance=return_distance)

    if return_distance:
        return knn[1], knn[0]
    return knn
