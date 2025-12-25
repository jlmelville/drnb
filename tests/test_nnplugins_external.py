import logging
import textwrap
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


def _success_spec(tmp_path: Path) -> NNPluginSpec:
    script_path = tmp_path / "nn_plugin_stub.py"
    script_path.write_text(
        textwrap.dedent(
            """\
            import argparse
            import json
            from pathlib import Path
            import sys

            import numpy as np

            parser = argparse.ArgumentParser()
            parser.add_argument("--method")
            parser.add_argument("--request", required=True)
            args = parser.parse_args()

            request = json.loads(Path(args.request).read_text(encoding="utf-8"))
            result_path = Path(request["output"]["result_path"])
            response_path = Path(request["output"]["response_path"])
            result_path.parent.mkdir(parents=True, exist_ok=True)

            x = np.load(request["input"]["x_path"], allow_pickle=False)
            n_neighbors = int(request["n_neighbors"])
            idx = np.zeros((x.shape[0], n_neighbors), dtype=np.int32)
            dist = np.zeros((x.shape[0], n_neighbors), dtype=np.float32)
            np.savez(result_path, idx=idx, dist=dist)

            response = {"ok": True, "result_npz": str(result_path)}
            response_path.write_text(json.dumps(response), encoding="utf-8")

            print("plugin stdout")
            print("plugin stderr", file=sys.stderr)
            """
        ),
        encoding="utf-8",
    )
    return NNPluginSpec(
        method="demo",
        plugin_dir=tmp_path,
        runner=["python", str(script_path)],
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


def _run_success_plugin(tmp_path: Path, quiet_logs: bool) -> None:
    run_external_neighbors(
        method="demo",
        spec=_success_spec(tmp_path),
        data=np.zeros((2, 2), dtype=np.float32),
        n_neighbors=1,
        metric="euclidean",
        params={},
        return_distance=True,
        quiet_plugin_logs=quiet_logs,
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


def test_plugin_logs_emitted_by_default(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        _run_success_plugin(tmp_path, quiet_logs=False)

    assert any("launching:" in record.getMessage() for record in caplog.records)
    stdout_logs = [
        record for record in caplog.records if record.name.endswith(".stdout")
    ]
    stderr_logs = [
        record for record in caplog.records if record.name.endswith(".stderr")
    ]
    assert stdout_logs
    assert stderr_logs


def test_plugin_logs_can_be_suppressed(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        _run_success_plugin(tmp_path, quiet_logs=True)

    assert not any("launching:" in record.getMessage() for record in caplog.records)
    assert not any(record.name.endswith(".stdout") for record in caplog.records)
    assert not any(record.name.endswith(".stderr") for record in caplog.records)
