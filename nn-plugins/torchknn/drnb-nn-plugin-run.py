#!/usr/bin/env python
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from drnb_nn_plugin_sdk.helpers.logging import log, summarize_params
from drnb_nn_plugin_sdk.helpers.paths import resolve_x_path
from drnb_nn_plugin_sdk.helpers.results import save_neighbors_npz
from drnb_nn_plugin_sdk.helpers.runner import run_nn_plugin
from drnb_nn_plugin_sdk.protocol import NNPluginRequest

TORCHKNN_METRICS = {
    "euclidean": "euclidean",
    "cosine": "cosine",
}

# Simple type aliases for clarity in type checkers / editors
ArrayF32 = npt.NDArray[np.float32]
ArrayI64 = npt.NDArray[np.int64]


def exact_knn_pytorch_batched(
    data: ArrayF32,
    k: int,
    batch_size: int,
    device: str | None = None,
    metric: str = "euclidean",
) -> tuple[ArrayI64, ArrayF32]:
    """
    Compute exact k-nearest neighbors with Euclidean or cosine distance using PyTorch.
    Note: indices[i, 0] will usually be i (self), but if there are exact duplicates,
    any of the zero-distance neighbors may appear first.

    Parameters
    ----------
    data : (N, D) float32
        One point per row.
    k : int
        Number of neighbors per point, INCLUDING the point itself.
        So indices[i, 0] == i and distances[i, 0] == 0.
    batch_size : int
        Number of query points per batch.
    device : {'cuda', 'mps', 'cpu', None}
        If None, auto-detects (MPS > CUDA > CPU).

    Returns
    -------
    indices : (N, k) int64
        Indices of k nearest neighbors for each point.
    distances : (N, k) float32
        Distances to those neighbors (Euclidean or cosine distance = 1 - cosine similarity).
    """
    if data.ndim != 2:
        raise ValueError("Input must be a 2D array")

    N = data.shape[0]
    if not (1 <= k <= N):
        log(f"[torchknn] k > N: k={k} N={N}, using k={N}")
        k = N

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    if metric not in TORCHKNN_METRICS:
        raise ValueError(f"Unsupported metric '{metric}' for torchknn")

    # ----- Choose device -----
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    log(f"[torchknn] Using device: {device}")

    # Move data to chosen device – all heavy ops will follow it there.
    x = torch.from_numpy(data.astype(np.float32, copy=False)).to(device)

    if metric == "cosine":
        x = F.normalize(x, p=2, dim=1, eps=1e-12)
    else:
        # Precompute ||x_j||^2 for all database points
        x_norm = (x**2).sum(dim=1)  # (N,)

    all_indices: list[torch.Tensor] = []
    all_distances: list[torch.Tensor] = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        # Query batch: (B, D) on device
        batch = x[start:end]

        if metric == "cosine":
            # Cosine similarity on normalized vectors; highest sim = closest
            sim = batch @ x.T
            batch_sim, batch_idx = torch.topk(
                sim, k=k, dim=1, largest=True, sorted=True
            )
            batch_dist = 1.0 - batch_sim.clamp(-1.0, 1.0)
            all_indices.append(batch_idx.cpu())
            all_distances.append(batch_dist.cpu())
        else:
            # ||q_i||^2 for queries, shape (B, 1)
            batch_norm = (batch**2).sum(dim=1, keepdim=True)

            # Pairwise squared distances: (B, N)
            # dist^2(q_i, x_j) = ||q_i||^2 + ||x_j||^2 - 2 <q_i, x_j>
            dist_sq = batch_norm + x_norm.unsqueeze(0) - 2.0 * (batch @ x.T)
            dist_sq.clamp_min_(0.0)

            batch_dist_sq, batch_idx = torch.topk(
                dist_sq, k=k, dim=1, largest=False, sorted=True
            )

            all_indices.append(batch_idx.cpu())
            all_distances.append(batch_dist_sq.sqrt_().cpu())

    indices_t = torch.cat(all_indices, dim=0)
    distances_t = torch.cat(all_distances, dim=0)

    indices: ArrayI64 = indices_t.numpy().astype(np.int64, copy=False)
    distances: ArrayF32 = distances_t.numpy().astype(np.float32, copy=False)
    return indices, distances


def _handler(req: NNPluginRequest) -> dict:
    params = req.params or {}
    log(f"[torchknn] params={summarize_params(params)}")

    x_path = resolve_x_path(req)
    X = np.load(x_path, allow_pickle=False)

    metric = req.metric
    n_neighbors = int(req.n_neighbors)

    if "batch_size" not in params:
        batch_size = 1024
        log(f"[torchknn] using default batch size: {batch_size}")
    else:
        batch_size = int(params["batch_size"])

    if metric not in TORCHKNN_METRICS:
        raise ValueError(f"Unsupported metric '{metric}' for torchknn")

    indices, distances = exact_knn_pytorch_batched(
        X, k=n_neighbors + 1, batch_size=batch_size, metric=metric
    )

    result = save_neighbors_npz(req.output.result_path, indices, distances)  # type: ignore[arg-type]
    return result


if __name__ == "__main__":
    run_nn_plugin({"torchknn": _handler}, description="drnb TorchKNN NN plugin")
