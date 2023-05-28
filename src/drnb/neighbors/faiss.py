import numpy as np

from drnb.preprocess import numpyfy
from drnb.log import log

FAISS_STATUS = dict(loaded=False, ok=False)
FAISS_METRICS = ["cosine", "euclidean"]
FAISS_DEFAULTS = {}


def load_faiss():
    try:
        # pylint:disable=global-variable-undefined
        global faiss
        faiss = __import__("faiss", globals(), locals())
        FAISS_STATUS["ok"] = True
        FAISS_STATUS["loaded"] = True
    except ImportError:
        FAISS_STATUS["ok"] = False
        FAISS_STATUS["loaded"] = True


def faiss_metrics():
    if not FAISS_STATUS["loaded"]:
        load_faiss()
    if FAISS_STATUS["ok"]:
        return FAISS_METRICS
    return []


def faiss_neighbors(
    X,
    n_neighbors=15,
    metric="euclidean",
    return_distance=True,
    use_gpu=True,
):
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
        distances, indices = faiss_neighbors_gpu(index_flat, X, n_neighbors)
    else:
        log.debug("using CPU")
        distances, indices = faiss_neighbors_cpu(index_flat, X, n_neighbors)

    if return_distance:
        if metric == "euclidean":
            distances = np.sqrt(distances)
        if metric == "cosine":
            distances = 1.0 - distances
        return indices, distances
    return indices


def faiss_neighbors_generic(index, data, n_neighbors):
    index.add(data)
    return index.search(data, n_neighbors)


def faiss_neighbors_cpu(index, data, n_neighbors):
    return faiss_neighbors_generic(index, data, n_neighbors)


def faiss_neighbors_gpu(index, data, n_neighbors):
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  # Move index to GPU
    distances_gpu, indices_gpu = faiss_neighbors_generic(gpu_index, data, n_neighbors)
    return distances_gpu.copy(), indices_gpu.copy()
