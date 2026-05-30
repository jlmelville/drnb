import warnings

import numpy as np

from drnb.embed.mmds import Mmds, Nmds


def _data() -> np.ndarray:
    return np.random.default_rng(42).normal(size=(12, 4))


def _assert_no_future_warnings(embedder: Mmds | Nmds) -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        coords = embedder.embed_impl(_data(), {"random_state": 42})

    assert coords.shape == (12, 2)
    assert not [
        warning for warning in captured if issubclass(warning.category, FutureWarning)
    ]


def test_metric_mds_uses_current_sklearn_api() -> None:
    _assert_no_future_warnings(Mmds(max_iter=2))


def test_non_metric_mds_uses_current_sklearn_api() -> None:
    _assert_no_future_warnings(Nmds(max_iter=2))
