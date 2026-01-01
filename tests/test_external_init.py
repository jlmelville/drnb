from pathlib import Path

import numpy as np
import pytest

from drnb.plugins import external
from drnb.plugins.external import (
    ExternalEmbedder,
    PluginSpec,
    PluginWorkspace,
    PluginWorkspaceError,
    _prepare_init_path,
)


def test_embed_impl_missing_init_path_retains_workspace(tmp_path, monkeypatch) -> None:
    ws_dir = tmp_path / "workspace"
    ws_dir.mkdir()

    def fake_mkdtemp(prefix: str) -> str:
        return str(ws_dir)

    monkeypatch.setattr(external.tempfile, "mkdtemp", fake_mkdtemp)

    class FakeRegistry:
        def lookup(self, method: str) -> PluginSpec:
            return PluginSpec(method, tmp_path)

    monkeypatch.setattr(external, "get_registry", lambda: FakeRegistry())

    embedder = ExternalEmbedder(method="demo", drnb_init="nonexistent-init-path")

    with pytest.raises(PluginWorkspaceError, match="init path not found"):
        embedder.embed_impl(np.zeros((2, 2), dtype=np.float32), params={}, ctx=None)

    # Workspace should be retained on failure for inspection.
    assert ws_dir.exists()


def test_prepare_init_path_copies_file(tmp_path) -> None:
    src = tmp_path / "init.npy"
    data = np.array([[1.0, 2.0]], dtype=np.float32)
    np.save(src, data)

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    target = _prepare_init_path(workspace, src)
    assert target is not None
    assert target.exists()
    loaded = np.load(target)
    np.testing.assert_array_equal(loaded, data)


def test_precomputed_init_written_to_workspace(tmp_path) -> None:
    workspace = PluginWorkspace(
        path=tmp_path / "workspace", remove_on_exit=True, method="demo"
    )
    workspace.path.mkdir()
    embedder = ExternalEmbedder(method="demo")
    embedder.precomputed_init = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    request, _ = embedder.build_plugin_workspace(
        workspace=workspace,
        x=np.zeros((2, 3), dtype=np.float32),
        params={},
        ctx=None,
        use_knn=False,
        use_sandbox=True,
        keep_tmp=True,
    )

    init_path = request.input.init_path
    assert init_path is not None
    assert Path(init_path).exists()
    loaded = np.load(init_path)
    np.testing.assert_array_equal(loaded, embedder.precomputed_init)
