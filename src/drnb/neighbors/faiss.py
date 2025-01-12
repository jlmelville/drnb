from typing import List

import numpy as np

from drnb.log import log
from drnb.preprocess import numpyfy

FAISS_STATUS = {"loaded": False, "ok": False}
FAISS_METRICS = ["cosine", "euclidean"]
FAISS_DEFAULTS = {}


def load_faiss():
    """Try to load the faiss library. If it is not available, set the status to False."""
    try:
        # pylint:disable=global-variable-undefined
        global faiss
        faiss = __import__("faiss", globals(), locals())
        FAISS_STATUS["ok"] = True
        FAISS_STATUS["loaded"] = True
    except ImportError:
        FAISS_STATUS["ok"] = False
        FAISS_STATUS["loaded"] = True


def faiss_metrics() -> List[str]:
    """Return the list of available metrics for the Faiss library."""
    if not FAISS_STATUS["loaded"]:
        load_faiss()
    if FAISS_STATUS["ok"]:
        return FAISS_METRICS
    return []


def faiss_neighbors(
    X: np.ndarray,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    return_distance: bool = True,
    use_gpu: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the nearest neighbors using the Faiss library. If `return_distance` is
    True, the function will return both the indices and the distances to the neighbors.
    Otherwise, it will only return the indices. If `use_gpu` is True, the function will
    use the GPU to compute the nearest neighbors."""
    if not FAISS_STATUS["loaded"]:
        load_faiss()
    if not FAISS_STATUS["ok"]:
        raise NotImplementedError("faiss not available")

    if metric == "cosine":
        faiss_space = faiss.IndexFlatIP
    elif metric == "euclidean":
        faiss_space = faiss.IndexFlatL2
    else:
        raise ValueError(f"Unsupported metric for faiss: '{metric}'")

    X = numpyfy(X, dtype=np.float32, layout="c")
    if metric == "cosine":
        faiss.normalize_L2(X)

    index_flat = faiss_space(X.shape[1])
    if use_gpu:
        log.debug("using GPU")
        distances, indices = _faiss_neighbors_gpu(index_flat, X, n_neighbors)
    else:
        log.debug("using CPU")
        distances, indices = _faiss_neighbors_cpu(index_flat, X, n_neighbors)

    if return_distance:
        if metric == "euclidean":
            distances = np.sqrt(distances)
        if metric == "cosine":
            distances = 1.0 - distances
        return indices, distances
    return indices


def _faiss_neighbors_generic(index, data, n_neighbors):
    index.add(data)
    return index.search(data, n_neighbors)


def _faiss_neighbors_cpu(index, data, n_neighbors):
    return _faiss_neighbors_generic(index, data, n_neighbors)


def _faiss_neighbors_gpu(index, data, n_neighbors):
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  # Move index to GPU
    distances_gpu, indices_gpu = _faiss_neighbors_generic(gpu_index, data, n_neighbors)
    return distances_gpu.copy(), indices_gpu.copy()
