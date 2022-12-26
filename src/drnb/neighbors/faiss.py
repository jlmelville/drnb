import numpy as np

from drnb.preprocess import numpyfy

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
):
    if not FAISS_STATUS["loaded"]:
        load_faiss()
    if not FAISS_STATUS["ok"]:
        # faiss = None
        raise NotImplementedError("faiss not available")

    if metric == "cosine":
        faiss_space = faiss.IndexFlatIP
    elif metric == "euclidean":
        faiss_space = faiss.IndexFlatL2
    else:
        raise ValueError(f"Unsupported metric for faiss: '{metric}'")

    X = numpyfy(X, dtype=np.float32, layout="c")

    res = faiss.StandardGpuResources()
    index_flat = faiss_space(X.shape[1])
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

    if metric == "cosine":
        faiss.normalize_L2(X)
    gpu_index_flat.add(X)

    distances, indices = gpu_index_flat.search(X, n_neighbors)
    if return_distance:
        if metric == "euclidean":
            distances = np.sqrt(distances)
        return indices, distances
    return indices
