import numpy as np

from drnb.io import numpyfy

FAISS_METRICS = ["cosine", "euclidean"]

FAISS_DEFAULTS = {}


def faiss_neighbors(
    X,
    n_neighbors=15,
    metric="euclidean",
    return_distance=True,
):
    # pylint: disable=import-outside-toplevel
    import faiss

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
        return indices, np.sqrt(distances)
    return indices
