from pathlib import Path

import numpy as np

from drnb_plugin_sdk.helpers.neighbors import (
    NbrInfo,
    NearestNeighbors,
    find_candidate_neighbors_info,
    read_neighbors,
    write_neighbors,
)


def test_find_candidate_neighbors_info(tmp_path: Path) -> None:
    base = tmp_path / "nn"
    base.mkdir()
    idx = np.arange(20, dtype=np.int32).reshape(10, 2)
    dist = np.arange(20, dtype=np.float32).reshape(10, 2)
    np.save(base / "toy.2.euclidean.exact.faiss.idx.npy", idx)
    np.save(base / "toy.2.euclidean.exact.faiss.dist.npy", dist)

    info = find_candidate_neighbors_info(
        base,
        "toy",
        n_neighbors=2,
        metric="euclidean",
        method=None,
        exact=True,
    )
    assert info is not None
    assert info.method == "faiss"
    assert info.n_nbrs == 2


def test_write_and_read_neighbors(tmp_path: Path) -> None:
    base = tmp_path / "nn"
    base.mkdir()
    idx = np.arange(20, dtype=np.int32).reshape(10, 2)
    dist = np.arange(20, dtype=np.float32).reshape(10, 2)
    info = NbrInfo(
        name="toy",
        n_nbrs=2,
        metric="euclidean",
        exact=True,
        method="faiss",
        has_distances=True,
        idx_path=None,
        dist_path=None,
    )
    nbrs = NearestNeighbors(idx=idx, dist=dist, info=info)
    write_neighbors(base, nbrs)

    loaded = read_neighbors(
        base,
        "toy",
        n_neighbors=2,
        metric="euclidean",
        exact=True,
    )
    assert loaded is not None
    assert loaded.idx.shape == (10, 2)
    assert np.allclose(loaded.dist, dist)
