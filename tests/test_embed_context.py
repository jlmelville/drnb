from pathlib import Path

import pytest

from drnb.embed.context import EmbedContext, read_neighbors_with_ctx


def test_read_neighbors_with_ctx_returns_none_without_context() -> None:
    assert read_neighbors_with_ctx("euclidean", 5, ctx=None) is None


def test_read_neighbors_with_ctx_uses_context(monkeypatch, tmp_path: Path) -> None:
    ctx = EmbedContext(
        dataset_name="digits",
        embed_method_name="pacmap",
        drnb_home=tmp_path,
        nn_sub_dir="custom-nn",
    )
    sentinel = object()
    captured: dict[str, object] = {}

    def fake_read_neighbors(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr("drnb.embed.context.read_neighbors", fake_read_neighbors)

    result = read_neighbors_with_ctx(
        metric="manhattan",
        n_neighbors=42,
        ctx=ctx,
        method="faiss",
        exact=True,
        return_distance=False,
        verbose=True,
    )

    assert result is sentinel
    assert captured["name"] == "digits"
    assert captured["n_neighbors"] == 42
    assert captured["metric"] == "manhattan"
    assert captured["method"] == "faiss"
    assert captured["exact"] is True
    assert captured["drnb_home"] == tmp_path
    assert captured["sub_dir"] == "custom-nn"
    assert captured["return_distance"] is False
    assert captured["verbose"] is True
