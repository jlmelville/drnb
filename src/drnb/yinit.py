import numpy as np
import openTSNE.initialization
import scipy.sparse.csgraph
import umap
from sklearn.utils import check_random_state

import drnb.neighbors.random
from drnb.log import log


def scale_coords(coords, max_coord=10.0):
    expansion = max_coord / np.abs(coords).max()
    return (coords * expansion).astype(np.float32)


def noisy_scale_coords(coords, max_coord=10.0, noise=0.0001, seed=None):
    return scale_coords(coords, max_coord=max_coord) + add_noise(
        coords, noise=noise, seed=seed
    )


def add_noise(coords, noise=0.0001, seed=None):
    rng = np.random.default_rng(seed=seed)
    return coords + rng.normal(scale=noise, size=coords.shape).astype(np.float32)


def spca(data):
    log.info("Initializing via openTSNE (scaled) PCA")
    return openTSNE.initialization.pca(data)


def gspectral(
    x, knn, metric="euclidean", op="intersection", weight=0.2, random_state=42
):

    randnn = drnb.neighbors.random.random_neighbors(
        x, distance=metric, random_state=random_state
    )
    rand_fss, _, _ = umap.umap_.fuzzy_simplicial_set(
        X=x,
        knn_indices=randnn.idx,
        knn_dists=randnn.dist,
        n_neighbors=randnn.idx.shape[1],
        random_state=None,
        metric=None,
    )

    knn_fss, _, _ = umap.umap_.fuzzy_simplicial_set(
        X=x,
        knn_indices=knn[0],
        knn_dists=knn[1],
        n_neighbors=knn[0].shape[1],
        random_state=None,
        metric=None,
    )

    log.info(
        "Creating random weighted graph with random n_neighbors = %d",
        randnn.idx.shape[1],
    )

    if op == "intersection":
        log.info("Combining sets by intersection with weight %.2f", weight)
        result = umap.umap_.general_simplicial_set_intersection(
            knn_fss, rand_fss, weight=weight
        )
        result = umap.umap_.reset_local_connectivity(result, reset_local_metric=True)
    elif op == "union":
        log.info("Combining sets by union")
        result = umap.umap_.general_simplicial_set_union(knn_fss, rand_fss)
        result = umap.umap_.reset_local_connectivity(result, reset_local_metric=True)
    elif op == "difference":
        log.info("Combining sets by difference with weight %.2f", weight)
        result = umap.umap_.general_simplicial_set_intersection(
            knn_fss, rand_fss, weight=weight, right_complement=True
        )
        # https://github.com/lmcinnes/umap/discussions/841
        # "resetting local connectivity there ... pretty much eliminated anything
        # __sub__ did, so I just didn't do it in that case."
        result = umap.umap_.reset_local_connectivity(result, reset_local_metric=False)
    elif op == "linear":
        result = weight * knn_fss + (1.0 - weight) * rand_fss
        result = umap.umap_.reset_local_connectivity(result, reset_local_metric=True)
    else:
        raise ValueError(f"Unknown set operation: '{op}'")

    nc = scipy.sparse.csgraph.connected_components(result)[0]
    if nc > 1:
        log.warning("global-weighted graph has %d components", nc)

    random_state = check_random_state(random_state)
    init = umap.umap_.spectral_layout(
        data=None,
        graph=result,
        dim=2,
        random_state=random_state,
    )

    return noisy_scale_coords(init, seed=random_state.randint(np.iinfo(np.uint32).max))
