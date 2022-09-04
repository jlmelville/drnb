from dataclasses import dataclass

import pacmap

import drnb.embed


@dataclass
class Pacmap(drnb.embed.Embedder):
    def embed_impl(self, x, params, ctx=None):
        return embed_pacmap(x, params)


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
def embed_pacmap(x, params):
    if "init" in params:
        init = params["init"]
        del params["init"]
    else:
        init = None

    embedder = pacmap.PaCMAP(**params)
    result = embedder.fit_transform(x, init=init)

    if params.get("intermediate", False):
        embedded = dict(coords=result[-1])
        for i in range(result.shape[0]):
            embedded[f"it_{embedder.intermediate_snapshots[i]}"] = result[i]
    else:
        embedded = result

    return embedded
