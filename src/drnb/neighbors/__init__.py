import itertools
from collections import defaultdict

import sklearn.metrics

import drnb.neighbors.sklearn as sknbrs
from drnb.io import numpyfy

from . import annoy, faiss, hnsw, pynndescent


def get_neighbors(
    data,
    n_neighbors=15,
    metric="euclidean",
    method="approximate",
    return_distance=True,
    verbose=False,
    method_kwds=None,
):
    if method in ("exact", "approximate"):
        if method == "exact":
            method = find_exact_method(metric)
        else:
            method = find_fast_method(metric)
    if (
        verbose
        and method == "sklearn"
        and data.shape[0] > 10000
        or data.shape[1] > 10000
    ):
        print("Using sklearn to find exact nearest neighbors: this might take a while")

    if method == "sklearn":
        nn_func = sknbrs.sklearn_neighbors
        default_method_kwds = sknbrs.SKLEARN_DEFAULTS
    elif method == "faiss":
        nn_func = faiss.faiss_neighbors
        default_method_kwds = faiss.FAISS_DEFAULTS
    elif method == "pynndescent":
        nn_func = pynndescent.pynndescent_neighbors
        default_method_kwds = pynndescent.PYNNDESCENT_DEFAULTS
    elif method == "hnsw":
        nn_func = hnsw.hnsw_neighbors
        default_method_kwds = hnsw.HNSW_DEFAULTS
    elif method == "annoy":
        nn_func = annoy.annoy_neighbors
        default_method_kwds = annoy.ANNOY_DEFAULTS
    else:
        raise ValueError(f"Unknown nearest neighbor method '{method}'")

    # dict merge operator Python 3.9+ only
    if method_kwds is None:
        method_kwds = default_method_kwds
    else:
        method_kwds = default_method_kwds | method_kwds

    if verbose:
        print(
            f"Finding {n_neighbors} neighbors using {method} "
            + f"with {metric} metric and params: {method_kwds}"
        )

    return nn_func(
        data,
        n_neighbors=n_neighbors,
        metric=metric,
        return_distance=return_distance,
        **method_kwds,
    )


def dmat(x):
    if isinstance(x, tuple) and len(x) == 2:
        x = x[0]
    x = numpyfy(x)
    return sklearn.metrics.pairwise_distances(x)


def zip_algs(metrics, method_name):
    return list(zip(metrics, itertools.cycle((method_name,))))


def find_exact_method(metric):
    return preferred_exact_methods()[metric][0]


def find_fast_method(metric):
    return preferred_fast_methods()[metric][0]


# For a given metric, return the suggested method for approximate nearest neighbors
# FAISS is included because it's fast
def preferred_fast_methods():
    metric_algs = defaultdict(list)
    for metric, method in (
        zip_algs(faiss.FAISS_METRICS.keys(), "faiss")
        + zip_algs(pynndescent.PYNNDESCENT_METRICS.keys(), "pynndescent")
        + zip_algs(hnsw.HNSW_METRICS.keys(), "hnsw")
        + zip_algs(annoy.ANNOY_METRICS.keys(), "annoy")
    ):
        metric_algs[metric].append(method)

    return metric_algs


def preferred_exact_methods():
    metric_algs = defaultdict(list)
    for metric, method in zip_algs(faiss.FAISS_METRICS.keys(), "faiss") + zip_algs(
        sknbrs.SKLEARN_METRICS.keys(), "sklearn"
    ):
        metric_algs[metric].append(method)

    return metric_algs
