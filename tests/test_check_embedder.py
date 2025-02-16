from drnb.embed import check_embed_method
from drnb.embed.pipeline import embedder


def test_check_embed_method():
    assert check_embed_method(
        embedder(
            "pacmap", local_scale=False, params=dict(n_neighbors=15, apply_pca=True)
        ),
        params=dict(apply_pca=False),
    ) == (
        "pacmap",
        {"local_scale": False, "params": {"n_neighbors": 15, "apply_pca": False}},
    )
