import numpy as np

from drnb.embed.context import EmbedContext
from drnb.embed.factory import create_embedder


def _toy_context(method_name: str) -> EmbedContext:
    return EmbedContext(dataset_name="toy", embed_method_name=method_name)


def test_pca_plugin_matches_inprocess() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 5)).astype(np.float32)
    params = {"random_state": 0}

    plugin = create_embedder(
        "pca-plugin",
        {"params": params, "use_precomputed_knn": False},
    )
    builtin = create_embedder("pca", {"params": params})

    plugin_res = plugin.embed(x, ctx=_toy_context("pca-plugin"))
    builtin_res = builtin.embed(x, ctx=_toy_context("pca"))

    plugin_coords = plugin_res["coords"]
    builtin_coords = (
        builtin_res["coords"] if isinstance(builtin_res, dict) else builtin_res
    )

    assert plugin_coords.shape == builtin_coords.shape == (40, 2)
    np.testing.assert_allclose(plugin_coords, builtin_coords, atol=1e-6)
