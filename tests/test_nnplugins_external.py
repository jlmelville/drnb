import logging
from pathlib import Path

import numpy as np
import pytest

from drnb.nnplugins.external import (
    NNPluginWorkspaceError,
    run_external_neighbors,
)
from drnb.nnplugins.registry import NNPluginSpec


def _failing_spec(tmp_path: Path) -> NNPluginSpec:
    return NNPluginSpec(
        method="demo",
        plugin_dir=tmp_path,
        runner=["python", "-c", "import sys; sys.stderr.write('fail\\n'); sys.exit(1)"],
    )


def _mk_workspace(monkeypatch, workspace: Path) -> None:
    workspace.mkdir()

    def fake_mkdtemp(prefix: str) -> str:
        return str(workspace)

    monkeypatch.setattr(
        "drnb.nnplugins.external.tempfile.mkdtemp", fake_mkdtemp, raising=True
    )


def _run_failing_plugin(tmp_path: Path, quiet: bool) -> None:
    run_external_neighbors(
        method="demo",
        spec=_failing_spec(tmp_path),
        data=np.zeros((2, 2), dtype=np.float32),
        n_neighbors=1,
        metric="euclidean",
        params={},
        return_distance=False,
        quiet_failures=quiet,
    )


def test_plugin_failure_retains_workspace_by_default(tmp_path, monkeypatch, caplog):
    workspace = tmp_path / "workspace-default"
    _mk_workspace(monkeypatch, workspace)
    with caplog.at_level(logging.INFO):
        with pytest.raises(NNPluginWorkspaceError):
            _run_failing_plugin(tmp_path, quiet=False)

    stderr_logs = [
        record for record in caplog.records if record.name.endswith(".stderr")
    ]
    assert stderr_logs
    assert any(record.levelno == logging.WARNING for record in stderr_logs)
    assert workspace.exists()


def test_plugin_failure_can_be_quiet(tmp_path, monkeypatch, caplog):
    workspace = tmp_path / "workspace-quiet"
    _mk_workspace(monkeypatch, workspace)

    with caplog.at_level(logging.INFO):
        with pytest.raises(NNPluginWorkspaceError):
            _run_failing_plugin(tmp_path, quiet=True)

    stderr_logs = [
        record for record in caplog.records if record.name.endswith(".stderr")
    ]
    assert stderr_logs
    assert all(record.levelno == logging.INFO for record in stderr_logs)
    assert not workspace.exists()
