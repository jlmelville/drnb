import numpy as np

from drnb.embed.context import EmbedContext
from drnb.embed.factory import create_embedder


def _ctx(name: str, drnb_home) -> EmbedContext:
    return EmbedContext(dataset_name="toy", embed_method_name=name, drnb_home=drnb_home)


def _assert_coords_close(a: np.ndarray, b: np.ndarray, atol: float = 1e-6) -> None:
    np.testing.assert_allclose(a, b, atol=atol)


def _assert_distance_structure_close(a: np.ndarray, b: np.ndarray, atol: float = 1e-4) -> None:
    dist_a = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1)
    dist_b = np.linalg.norm(b[:, None, :] - b[None, :, :], axis=-1)
    if dist_a.max() > 0:
        dist_a = dist_a / dist_a.max()
    if dist_b.max() > 0:
        dist_b = dist_b / dist_b.max()
    np.testing.assert_allclose(dist_a, dist_b, atol=atol)


def test_tsne_plugin_matches_inprocess(tmp_path) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((30, 5)).astype(np.float32)
    params = {
        "perplexity": 10,
        "learning_rate": 100,
        "random_state": 0,
        "n_iter": 250,
    }

    plugin = create_embedder(
        "tsne-plugin",
        {"params": params, "use_precomputed_knn": True},
    )
    builtin = create_embedder(
        "tsne",
        {"params": dict(params), "use_precomputed_knn": True},
    )

    ctx_plugin = _ctx("tsne-plugin", tmp_path)
    ctx_builtin = _ctx("tsne", tmp_path)

    plugin_res = plugin.embed(x, ctx=ctx_plugin)
    builtin_res = builtin.embed(x, ctx=ctx_builtin)

    plugin_coords = plugin_res["coords"]
    builtin_coords = (
        builtin_res["coords"] if isinstance(builtin_res, dict) else builtin_res
    )

    assert plugin_coords.shape == builtin_coords.shape == (30, 2)
    _assert_coords_close(plugin_coords, builtin_coords)


def test_tsne_plugin_annealed_exaggeration(tmp_path) -> None:
    rng = np.random.default_rng(42)
    x = rng.standard_normal((25, 4)).astype(np.float32)
    params = {
        "perplexity": 8,
        "random_state": 0,
        "n_exaggeration_iter": 50,
        "n_anneal_steps": 50,
        "n_iter": 100,
        "anneal_exaggeration": True,
    }

    plugin = create_embedder(
        "tsne-plugin",
        {"params": params, "use_precomputed_knn": True},
    )
    builtin_params = dict(params)
    builtin_params.pop("anneal_exaggeration", None)
    builtin = create_embedder(
        "tsne",
        {
            "params": builtin_params,
            "use_precomputed_knn": True,
            "anneal_exaggeration": True,
        },
    )

    ctx_plugin = _ctx("tsne-plugin", tmp_path)
    ctx_builtin = _ctx("tsne", tmp_path)

    plugin_res = plugin.embed(x, ctx=ctx_plugin)
    builtin_res = builtin.embed(x, ctx=ctx_builtin)

    plugin_coords = plugin_res["coords"]
    builtin_coords = (
        builtin_res["coords"] if isinstance(builtin_res, dict) else builtin_res
    )

    assert plugin_coords.shape == builtin_coords.shape == (25, 2)
    _assert_distance_structure_close(plugin_coords, builtin_coords)
