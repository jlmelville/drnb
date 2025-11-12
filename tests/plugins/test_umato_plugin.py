import numpy as np

from drnb.embed.context import EmbedContext
from drnb.embed.factory import create_embedder


def _ctx(name: str, drnb_home) -> EmbedContext:
    return EmbedContext(dataset_name="toy", embed_method_name=name, drnb_home=drnb_home)


def _assert_contract(result, expected_rows: int) -> None:
    coords = result["coords"]
    assert coords.shape == (expected_rows, 2)
    assert np.isfinite(coords).all()


def test_umato_plugin_basic(tmp_path) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((60, 12)).astype(np.float32)

    embedder = create_embedder("umato-plugin", {"params": {"random_state": 0}})
    result = embedder.embed(x, ctx=_ctx("umato-plugin", tmp_path))

    _assert_contract(result, expected_rows=60)
