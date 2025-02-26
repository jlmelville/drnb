from dataclasses import dataclass

import numpy as np
import pacmap
import pandas as pd

import drnb.embed
import drnb.embed.base
from drnb.embed.context import EmbedContext, get_neighbors_with_ctx
from drnb.log import log
from drnb.neighbors.localscale import locally_scaled_neighbors
from drnb.types import EmbedResult


# See also:
# https://github.com/YingfanWang/PaCMAP/blob/master/demo/specify_nn_demo.py
def create_neighbor_pairs(nbr_idx: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Create pairs of neighbor indices from k-nearest neighbor indices.

    Parameters
    ----------
    nbr_idx : np.ndarray
        (N, k) array of neighbor indices. The first column (nbr_idx[:, 0]) is expected
        to contain self-neighbors (i.e., nbr_idx[i, 0] == i for all i).
    n_neighbors : int
        Number of neighbors to use for creating pairs. Must be <= nbr_idx.shape[1].
        If nbr_idx includes self-neighbors, n_neighbors should typically be
        nbr_idx.shape[1] - 1 to exclude self-pairs.

    Returns
    -------
    np.ndarray
        (N * n_neighbors, 2) array of neighbor pairs. Each row contains [i, j] where i
        is the source point and j is one of its neighbors. Self-pairs (where i == j)
        are not included in the output, even if present in the input nbr_idx.

    Raises
    ------
    ValueError
        If n_neighbors is greater than the number of columns in nbr_idx.
    """
    if n_neighbors > nbr_idx.shape[1]:
        raise ValueError(
            f"n_neighbors ({n_neighbors}) must be <= number of columns in "
            f"nbr_idx ({nbr_idx.shape[1]})"
        )

    n = len(nbr_idx)
    pairs = np.zeros((n * n_neighbors, 2), dtype=np.int32)
    for i in range(n):
        for j in range(n_neighbors):
            pairs[i * n_neighbors + j] = [i, nbr_idx[i, j + 1]]
    return pairs


@dataclass
class Pacmap(drnb.embed.base.Embedder):
    """PaCMAP embedder.

    Attributes:
        use_precomputed_knn: Whether to use precomputed knn.
        init: Initialization method, one of "pca", "random" or user-supplied.
        local_scale: Whether to apply local scaling to precomputed neighbors.
        local_scale_kwargs: Optional kwargs for local scaling (l, m, scale_from,
            scale_to).

    Possible params:
        distance="euclidean" (str): Distance metric.
        n_neighbors=10 (int): Number of neighbors.
        MN_ratio=0.5 (float): Ratio of mid near pairs to nearest neighbor pairs.
        FP_ratio=2.0 (float): Ratio of further pairs to nearest neighbor pairs.
        lr=1.0 (float): Learning rate of the Adam optimizer.
        num_iters=450 (int): Number of iterations.
        apply_pca=True (bool): Whether to apply PCA on the input data.
        intermediate=False (bool): Whether to return intermediate snapshots.
        intermediate_snapshots (list): Iterations at which snapshots are taken.
        random_state=None (int): Random seed.
    """

    use_precomputed_knn: bool = True
    init: str = None
    local_scale: bool = True
    local_scale_kwargs: dict | None = None

    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> EmbedResult:
        knn_params = {}
        if isinstance(self.use_precomputed_knn, dict):
            knn_params = dict(self.use_precomputed_knn)
            self.use_precomputed_knn = True

        apply_pca = params.get("apply_pca", True)
        if self.use_precomputed_knn:
            # if we aren't applying PCA or we are but the data is less than 100D, we can
            # use the precomputed knn
            if not apply_pca or x.shape[1] <= 100:
                log.info("Using precomputed knn")
                metric = params.get("distance", "euclidean")
                # Plus 1 to account for the self-neighbor
                n_neighbors = params.get("n_neighbors", 10) + 1

                scale_kwargs = {}
                if self.local_scale:
                    # Default local scaling parameters
                    # we already accounted for self-neighbor in n_neighbors so we add 50
                    # to the number of neighbors to scale
                    scale_kwargs = {
                        "l": n_neighbors,
                        "m": n_neighbors + 50,
                        "scale_from": 4,
                        "scale_to": 6,
                    }
                    if self.local_scale_kwargs is not None:
                        scale_kwargs.update(self.local_scale_kwargs)
                    knn_neighbors = scale_kwargs["m"]
                else:
                    knn_neighbors = n_neighbors

                precomputed_knn = get_neighbors_with_ctx(
                    x, metric, knn_neighbors, knn_params=knn_params, ctx=ctx
                )
                idx = precomputed_knn.idx

                if self.local_scale:
                    log.info(
                        "Applying local scaling to neighbors with params: %s",
                        scale_kwargs,
                    )

                    idx, _ = locally_scaled_neighbors(
                        idx, precomputed_knn.dist, **scale_kwargs
                    )

                pair_neighbors = create_neighbor_pairs(idx, n_neighbors - 1)

                log.info(
                    "Converted knn to pair neighbors: %s",
                    pair_neighbors.shape,
                )
                params["pair_neighbors"] = pair_neighbors
            else:
                # otherwise, we are applying PCA and it will reduce the dimensionality
                # which can perturb the nearest neighbors so we can't use the
                # precomputed knn
                log.warning(
                    "Precomputed knn cannot be used: dimensionality will be reduced"
                    + " from %d to 100 dimensions.",
                    x.shape[1],
                )

        return embed_pacmap(x, params, self.init)


# n_neighbors=10
# MN_ratio=0.5 Ratio of mid near pairs to nearest neighbor pairs (e.g. n_neighbors=10, MN_ratio=0.5 --> 5 Mid near pairs).
# FP_ratio=2.0 Ratio of further pairs to nearest neighbor pairs (e.g. n_neighbors=10, FP_ratio=2 --> 20 Further pairs).
# pair_neighbors=None: numpy.ndarray of shape (X.shape[0] * n_neighbors, 2), Pre-calculated nearest neighbor pairs. There will be n_neighbors pairs per item i, of the form [i, j] where j is the index of the neighbors.
# pair_MN=None: numpy.ndarray of shape (X.shape[0] * n_mid_near, 2). Pre-calculated mid near pairs.
# pair_FP=None: numpy.ndarray of shape (X.shape[0] * n_further_pair, 2). Pre-calculated further pairs.
# distance="euclidean": distance metric. One of: "euclidean", "manhattan", "angular", "hamming".
# lr=1.0: learning rate of the Adam optimizer.
# num_iters=450. Number of iterations (epochs in UMAP-speak). Internally, different weights are used for the different types of pairs based on the absolute value of the iteration number (transitions at 100 and 200 iterations), so it is recommended to set this > 250.
# apply_pca=True: whether to apply PCA on the input data. Ignored if distance="hamming" or there are fewer than 100 dimensions in the input data. Otherwise, the first 100 components from truncated SVD are extracted. Data is centered. If no PCA is applied then data is scaled to 0-1 globally (columns maintain their ratio of variances) and then mean-centered.
# intermediate=False: if True, then snapshots of the coordinates at intermediate steps of the iteration are also returned.
# intermediate_snapshots=[0, 10, 30, 60, 100, 120, 140, 170, 200, 250, 300, 350, 450]: the iterations at which snapshots are taken. Ignored unless intermediate=True.
# random_state=None.
# init: one of "pca", "random" or user-supplied
def embed_pacmap(
    x: np.ndarray, params: dict, init: np.ndarray | str | None = None
) -> np.ndarray | dict:
    """Embed data using PaCMAP."""
    # if intermediate_snapshots is supplied, make sure the last value is `num_iters`
    # to avoid an exception
    if params.get("intermediate", False) and "intermediate_snapshots" in params:
        snapshots = params["intermediate_snapshots"]
        num_iters = params.get("num_iters", 450)
        if snapshots[-1] != num_iters:
            snapshots.append(num_iters)
        params["intermediate_snapshots"] = snapshots

    log.info("Running PaCMAP")
    embedder = pacmap.PaCMAP(**params)
    result = embedder.fit_transform(x, init=init)
    log.info("Embedding completed")

    if params.get("intermediate", False):
        embedded = {"coords": result[-1], "snapshots": {}}
        for i in range(result.shape[0]):
            embedded["snapshots"][f"it_{embedder.intermediate_snapshots[i]}"] = result[
                i
            ]
    else:
        embedded = result

    return embedded


@dataclass
class Localmap(drnb.embed.base.Embedder):
    """LocalMAP embedder.

    Attributes:
        use_precomputed_knn: Whether to use precomputed knn.
        init: Initialization method, one of "pca", "random" or user-supplied.
        local_scale: Whether to apply local scaling to precomputed neighbors.
        local_scale_kwargs: Optional kwargs for local scaling (l, m, scale_from,
            scale_to).
        low_dist_thres: The Proximal Cluster Distance Commons threshold for local FP.

    Possible params:
        distance="euclidean" (str): Distance metric.
        n_neighbors=10 (int): Number of neighbors.
        MN_ratio=0.5 (float): Ratio of mid near pairs to nearest neighbor pairs.
        FP_ratio=2.0 (float): Ratio of further pairs to nearest neighbor pairs.
        lr=1.0 (float): Learning rate of the Adam optimizer.
        num_iters=450 (int): Number of iterations.
        apply_pca=True (bool): Whether to apply PCA on the input data.
        intermediate=False (bool): Whether to return intermediate snapshots.
        intermediate_snapshots (list): Iterations at which snapshots are taken.
        random_state=None (int): Random seed.
        low_dist_thres=10 (float): Distance threshold for local FP.
    """

    use_precomputed_knn: bool = True
    init: str = None
    local_scale: bool = True
    local_scale_kwargs: dict | None = None
    low_dist_thres: float = 10.0

    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> EmbedResult:
        knn_params = {}
        if isinstance(self.use_precomputed_knn, dict):
            knn_params = dict(self.use_precomputed_knn)
            self.use_precomputed_knn = True

        apply_pca = params.get("apply_pca", True)
        if self.use_precomputed_knn:
            # if we aren't applying PCA or we are but the data is less than 100D, we can
            # use the precomputed knn
            if not apply_pca or x.shape[1] <= 100:
                log.info("Using precomputed knn")
                metric = params.get("distance", "euclidean")
                # Plus 1 to account for the self-neighbor
                n_neighbors = params.get("n_neighbors", 10) + 1

                scale_kwargs = {}
                if self.local_scale:
                    # Default local scaling parameters
                    # we already accounted for self-neighbor in n_neighbors so we add 50
                    # to the number of neighbors to scale
                    scale_kwargs = {
                        "l": n_neighbors,
                        "m": n_neighbors + 50,
                        "scale_from": 4,
                        "scale_to": 6,
                    }
                    if self.local_scale_kwargs is not None:
                        scale_kwargs.update(self.local_scale_kwargs)
                    knn_neighbors = scale_kwargs["m"]
                else:
                    knn_neighbors = n_neighbors

                precomputed_knn = get_neighbors_with_ctx(
                    x, metric, knn_neighbors, knn_params=knn_params, ctx=ctx
                )
                idx = precomputed_knn.idx

                if self.local_scale:
                    log.info(
                        "Applying local scaling to neighbors with params: %s",
                        scale_kwargs,
                    )

                    idx, _ = locally_scaled_neighbors(
                        idx, precomputed_knn.dist, **scale_kwargs
                    )

                pair_neighbors = create_neighbor_pairs(idx, n_neighbors - 1)

                log.info(
                    "Converted knn to pair neighbors: %s",
                    pair_neighbors.shape,
                )
                params["pair_neighbors"] = pair_neighbors
            else:
                # otherwise, we are applying PCA and it will reduce the dimensionality
                # which can perturb the nearest neighbors so we can't use the
                # precomputed knn
                log.warning(
                    "Precomputed knn cannot be used: dimensionality will be reduced"
                    + " from %d to 100 dimensions.",
                    x.shape[1],
                )

        # Add the low_dist_thres parameter to params
        params["low_dist_thres"] = self.low_dist_thres

        return embed_localmap(x, params, self.init)


def embed_localmap(
    x: np.ndarray, params: dict, init: np.ndarray | str | None = None
) -> np.ndarray | dict:
    """Embed data using LocalMAP."""
    # if intermediate_snapshots is supplied, make sure the last value is `num_iters`
    # to avoid an exception
    if params.get("intermediate", False) and "intermediate_snapshots" in params:
        snapshots = params["intermediate_snapshots"]
        num_iters = params.get("num_iters", 450)
        if snapshots[-1] != num_iters:
            snapshots.append(num_iters)
        params["intermediate_snapshots"] = snapshots

    log.info("Running LocalMAP")
    embedder = pacmap.LocalMAP(**params)
    result = embedder.fit_transform(x, init=init)
    log.info("Embedding completed")

    if params.get("intermediate", False):
        embedded = {"coords": result[-1], "snapshots": {}}
        for i in range(result.shape[0]):
            embedded["snapshots"][f"it_{embedder.intermediate_snapshots[i]}"] = result[
                i
            ]
    else:
        embedded = result

    return embedded
