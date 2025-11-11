import numpy as np

from drnb.embed.context import EmbedContext
from drnb.embed.factory import create_embedder


def _ctx(name: str, drnb_home) -> EmbedContext:
    return EmbedContext(dataset_name="toy", embed_method_name=name, drnb_home=drnb_home)


def _assert_snapshot_contract(
    result, n_rows: int, requested: list[int], final_iter: int
) -> None:
    coords = result["coords"]
    assert coords.shape == (n_rows, 2)
    assert np.isfinite(coords).all()

    snaps = result.get("snapshots") or {}
    expected_iters = sorted({*requested, final_iter})
    expected_keys = [f"it_{it}" for it in expected_iters]
    assert list(snaps.keys()) == expected_keys
    for key in expected_keys:
        snap = snaps[key]
        assert snap.shape == (n_rows, 2)
        assert np.isfinite(snap).all()


def test_pacmap_plugin_returns_snapshots(tmp_path) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 12)).astype(np.float32)
    snapshots = [0, 10, 30]
    params = {
        "n_neighbors": 8,
        "random_state": 0,
        "num_iters": (100, 100, 250),
        "apply_pca": False,
        "intermediate": True,
        "intermediate_snapshots": snapshots,
    }

    plugin = create_embedder(
        "pacmap-plugin",
        {
            "params": params,
            "init": "pca",
            "local_scale": False,
        },
    )

    plugin_res = plugin.embed(x, ctx=_ctx("pacmap-plugin", tmp_path))
    assert isinstance(plugin_res, dict)
    _assert_snapshot_contract(
        plugin_res, n_rows=40, requested=snapshots, final_iter=sum(params["num_iters"])
    )


def test_localmap_plugin_returns_snapshots(tmp_path) -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((35, 10)).astype(np.float32)
    snapshots = [0, 5, 25]
    params = {
        "n_neighbors": 6,
        "random_state": 0,
        "num_iters": (100, 100, 250),
        "apply_pca": False,
        "intermediate": True,
        "intermediate_snapshots": snapshots,
    }

    plugin = create_embedder(
        "localmap-plugin",
        {
            "params": params,
            "init": "pca",
            "local_scale": False,
        },
    )

    plugin_res = plugin.embed(x, ctx=_ctx("localmap-plugin", tmp_path))
    assert isinstance(plugin_res, dict)
    _assert_snapshot_contract(
        plugin_res, n_rows=35, requested=snapshots, final_iter=sum(params["num_iters"])
    )
