import numpy as np

from drnb.embed.context import EmbedContext
from drnb.embed.factory import create_embedder


def _ctx(name: str, drnb_home) -> EmbedContext:
    return EmbedContext(dataset_name="toy", embed_method_name=name, drnb_home=drnb_home)


def _assert_basic_contract(result, n_samples: int, n_components: int = 2) -> None:
    coords = result["coords"]
    assert coords.shape == (n_samples, n_components)
    assert np.isfinite(coords).all()


def test_pca_plugin_basic_contract(tmp_path) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 5)).astype(np.float32)
    embedder = create_embedder(
        "pca-plugin",
        {"params": {"random_state": 0}},
    )
    result = embedder.embed(x, ctx=_ctx("pca-plugin", tmp_path))
    _assert_basic_contract(result, n_samples=20)


def test_randproj_plugin_basic_contract(tmp_path) -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((25, 10)).astype(np.float32)
    embedder = create_embedder(
        "randproj-plugin",
        {"params": {"random_state": 0}},
    )
    result = embedder.embed(x, ctx=_ctx("randproj-plugin", tmp_path))
    _assert_basic_contract(result, n_samples=25)


def test_isomap_plugin_basic_contract(tmp_path) -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal((30, 6)).astype(np.float32)
    embedder = create_embedder(
        "isomap-plugin",
        {"params": {"n_neighbors": 8}},
    )
    result = embedder.embed(x, ctx=_ctx("isomap-plugin", tmp_path))
    _assert_basic_contract(result, n_samples=30)


def test_mmds_plugin_basic_contract(tmp_path) -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal((18, 7)).astype(np.float32)
    embedder = create_embedder(
        "mmds-plugin",
        {"params": {"random_state": 0, "max_iter": 20}},
    )
    result = embedder.embed(x, ctx=_ctx("mmds-plugin", tmp_path))
    _assert_basic_contract(result, n_samples=18)


def test_nmds_plugin_basic_contract(tmp_path) -> None:
    rng = np.random.default_rng(4)
    x = rng.standard_normal((18, 7)).astype(np.float32)
    embedder = create_embedder(
        "nmds-plugin",
        {"params": {"random_state": 0, "max_iter": 20}},
    )
    result = embedder.embed(x, ctx=_ctx("nmds-plugin", tmp_path))
    _assert_basic_contract(result, n_samples=18)
