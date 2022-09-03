import faiss
import numpy as np

FAISS_METRICS = {"cosine": faiss.IndexFlatIP, "euclidean": faiss.IndexFlatL2}

FAISS_DEFAULTS = {}


def faiss_neighbors(
    X,
    n_neighbors=15,
    metric="euclidean",
    return_distance=True,
):

    faiss_space = FAISS_METRICS[metric]

    X = np.ascontiguousarray(X.astype(np.float32))

    res = faiss.StandardGpuResources()
    index_flat = faiss_space(X.shape[1])
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

    if metric == "cosine":
        faiss.normalize_L2(X)
    gpu_index_flat.add(X)

    distances, indices = gpu_index_flat.search(X, n_neighbors)
    if return_distance:
        return indices, np.sqrt(distances)
    return indices
