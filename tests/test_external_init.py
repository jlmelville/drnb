from pathlib import Path

import numpy as np
import pytest

from drnb.plugins import external
from drnb.plugins.external import ExternalEmbedder, PluginSpec, _prepare_init_path


def test_embed_impl_missing_init_path_retains_workspace(tmp_path, monkeypatch) -> None:
    ws_dir = tmp_path / "workspace"
    ws_dir.mkdir()

    def fake_mkdtemp(prefix: str) -> str:
        return str(ws_dir)

    monkeypatch.setattr(external.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(external, "plugins_enabled", lambda: True)

    class FakeRegistry:
        def lookup(self, method: str) -> PluginSpec:
            return PluginSpec(method, tmp_path)

    monkeypatch.setattr(external, "get_registry", lambda: FakeRegistry())

    embedder = ExternalEmbedder(method="demo", drnb_init="nonexistent-init-path")

    with pytest.raises(RuntimeError, match="init path not found"):
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
