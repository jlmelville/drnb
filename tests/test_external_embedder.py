import numpy as np
import pytest

from drnb.plugins import external
from drnb.plugins.external import (
    ExternalEmbedder,
    PluginNeighbors,
    PluginWorkspace,
    PluginWorkspaceError,
    _prepare_neighbor_paths,
)


def test_plugin_workspace_fail_sets_flag(tmp_path) -> None:
    workspace = PluginWorkspace(path=tmp_path, remove_on_exit=True, method="demo")

    with pytest.raises(PluginWorkspaceError):
        workspace.fail("boom")

    assert workspace.remove_on_exit is False


def test_prepare_neighbor_paths_missing_neighbors(monkeypatch, tmp_path) -> None:
    def missing_neighbors(*args, **kwargs):
        raise FileNotFoundError("not found")

    monkeypatch.setattr(
        "drnb.embed.context.get_neighbors_with_ctx", missing_neighbors, raising=False
    )

    neighbors = _prepare_neighbor_paths(
        tmp_path,
        use_knn=True,
        source_neighbors=None,
        params={},
        x=np.zeros((2, 2), dtype=np.float32),
        ctx=None,
        method="demo",
    )

    assert neighbors == PluginNeighbors()


def test_prepare_neighbor_paths_unexpected_error(monkeypatch, tmp_path) -> None:
    def explode(*args, **kwargs):
        raise RuntimeError("bad knn")

    monkeypatch.setattr(
        "drnb.embed.context.get_neighbors_with_ctx", explode, raising=False
    )

    with pytest.raises(RuntimeError):
        _prepare_neighbor_paths(
            tmp_path,
            use_knn=True,
            source_neighbors=None,
            params={},
            x=np.zeros((2, 2), dtype=np.float32),
            ctx=None,
            method="demo",
        )


def test_external_embed_uses_param_copy(monkeypatch) -> None:
    embedder = ExternalEmbedder(method="demo")
    embedder.params = {"a": 1}

    def fake_impl(self, x, params, ctx=None):
        params["a"] = 2
        return {"coords": np.zeros((1, 1), dtype=np.float32)}

    monkeypatch.setattr(
        embedder, "embed_impl", fake_impl.__get__(embedder, ExternalEmbedder)
    )

    result = embedder.embed(np.zeros((1, 1), dtype=np.float32))

    assert embedder.params == {"a": 1}
    assert isinstance(result, dict)
    assert "coords" in result


def test_decode_plugin_result_captures_version(tmp_path) -> None:
    embedder = ExternalEmbedder(method="demo")
    coords = np.zeros((2, 2), dtype=np.float32)
    npz_path = tmp_path / "result.npz"
    np.savez_compressed(npz_path, coords=coords)

    response = {
        "ok": True,
        "result_npz": str(npz_path),
        "version": {
            "package": "demo-lib",
            "version": "1.0.0",
            "plugin_package": "drnb-plugin-demo",
            "plugin_version": "0.0.1",
        },
    }

    result = embedder.decode_plugin_result(
        workspace=PluginWorkspace(path=tmp_path, remove_on_exit=True, method="demo"),
        request=None,
        response=response,
    )
    assert result["version_info"]["source"] == "plugin"
    assert result["version_info"]["package"] == "demo-lib"
    assert result["version_info"]["plugin_package"] == "drnb-plugin-demo"
