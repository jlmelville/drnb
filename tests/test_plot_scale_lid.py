import numpy as np
from sklearn.neighbors import NearestNeighbors

from drnb.dimension import mle_local
from drnb.plot.scale.lid import ColorByLid


def _lid_expected(data: np.ndarray, n_neighbors: int, remove_self: bool) -> np.ndarray:
    n_neighbors_adj = min(n_neighbors + (1 if remove_self else 0), data.shape[0])
    distances, _ = (
        NearestNeighbors(n_neighbors=n_neighbors_adj).fit(data).kneighbors(data)
    )
    effective_neighbors = min(n_neighbors, data.shape[0] - (1 if remove_self else 0))
    return mle_local(
        distances,
        n_neighbors=effective_neighbors,
        remove_self=remove_self,
    )


def test_color_by_lid_matches_direct_mle():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(60, 5))
    coords = rng.normal(size=(60, 2))

    color_by = ColorByLid(n_neighbors=10, metric="euclidean")
    lid_values = color_by(data, None, coords, ctx=None)

    expected = _lid_expected(data, n_neighbors=10, remove_self=True)
    np.testing.assert_allclose(lid_values, expected)


def test_color_by_lid_clamps_neighbor_count():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(5, 3))
    coords = rng.normal(size=(5, 2))

    color_by = ColorByLid(n_neighbors=10, metric="euclidean")
    lid_values = color_by(data, None, coords, ctx=None)

    expected = _lid_expected(data, n_neighbors=4, remove_self=True)
    np.testing.assert_allclose(lid_values, expected)
