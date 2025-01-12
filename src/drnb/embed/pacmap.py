from dataclasses import dataclass

import numpy as np
import pacmap
import pandas as pd

import drnb.embed
import drnb.embed.base
from drnb.embed.context import EmbedContext, get_neighbors_with_ctx
from drnb.log import log
from drnb.types import EmbedResult


@dataclass
class Pacmap(drnb.embed.base.Embedder):
    """PaCMAP embedder.

    Attributes:
        use_precomputed_knn: Whether to use precomputed knn.
        init: Initialization method, one of "pca", "random" or user-supplied.

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

    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> EmbedResult:
        knn_params = {}
        if isinstance(self.use_precomputed_knn, dict):
            knn_params = dict(self.use_precomputed_knn)
            self.use_precomputed_knn = True

        if self.use_precomputed_knn:
            log.info("Using precomputed knn")
            metric = params.get("distance", "euclidean")
            n_neighbors = params.get("n_neighbors", 10) + 1
            precomputed_knn = get_neighbors_with_ctx(
                x, metric, n_neighbors, knn_params=knn_params, ctx=ctx
            )
            pair_neighbors = (
                pd.melt(pd.DataFrame(precomputed_knn.idx), [0])[[0, "value"]]
                .to_numpy()
                .astype(np.int32)
            )
            log.info(
                "Converted knn to pair neighbors: %s",
                pair_neighbors.shape,
            )
            params["pair_neighbors"] = pair_neighbors

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
