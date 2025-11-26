from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
from drnb_nn_plugin_sdk import (
    NN_PLUGIN_PROTOCOL_VERSION,
    NNPluginInputPaths,
    NNPluginOptions,
    NNPluginOutputPaths,
    NNPluginRequest,
    env_flag,
    request_to_dict,
)

from drnb.log import log
from drnb.neighbors.nbrinfo import NbrInfo, NearestNeighbors
from drnb.nnplugins.registry import NNPluginSpec
from drnb.util import FromDict


class NNPluginWorkspaceError(RuntimeError):
    """Error raised for NN plugin workspace failures."""


@dataclass
class NNPluginWorkspace:
    path: Path
    remove_on_exit: bool
    method: str
    retain_on_error: bool = True

    @property
    def prefix(self) -> str:
        return f"[nn-plugin:{self.method}]"

    def fail(self, message: str) -> NoReturn:
        """Raise with context and retain the workspace even if remove_on_exit is True."""
        if self.retain_on_error:
            self.remove_on_exit = False
            log.error(
                "%s failure occurred; workspace retained at %s",
                self.prefix,
                self.path,
            )
        else:
            log.info("%s failure occurred", self.prefix)
        raise NNPluginWorkspaceError(
            f"{self.prefix} {message} (workspace: {self.path})"
        )

    def __enter__(self) -> "NNPluginWorkspace":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None and self.remove_on_exit and self.retain_on_error:
            self.remove_on_exit = False
            log.error(
                "%s unexpected error; workspace retained at %s",
                self.prefix,
                self.path,
            )
        if self.remove_on_exit:
            shutil.rmtree(self.path, ignore_errors=True)
        else:
            log.info("%s kept plugin workspace at %s", self.prefix, self.path)


@dataclass
class NNPluginContextInfo(FromDict):
    dataset_name: str | None = None
    drnb_home: Path | None = None
    data_sub_dir: str = "data"
    nn_sub_dir: str = "nn"
    experiment_name: str | None = None


def run_external_neighbors(
    *,
    method: str,
    spec: NNPluginSpec,
    data: np.ndarray,
    n_neighbors: int,
    metric: str,
    params: dict[str, Any],
    return_distance: bool = True,
    ctx: NNPluginContextInfo | None = None,
    neighbor_name: str | None = None,
    quiet_failures: bool = False,
) -> NearestNeighbors:
    keep_tmp = env_flag("DRNB_NN_PLUGIN_KEEP_TMP", False)
    use_sandbox = env_flag("DRNB_NN_PLUGIN_SANDBOX_INPUTS", False)
    workspace = NNPluginWorkspace(
        path=Path(tempfile.mkdtemp(prefix=f"drnb-nn-{method}-")),
        remove_on_exit=not keep_tmp,
        method=method,
        retain_on_error=not quiet_failures,
    )
    stderr_level = logging.INFO if quiet_failures else logging.WARNING
    with workspace:
        request, req_path = _build_request(
            workspace=workspace,
            data=data,
            method=method,
            metric=metric,
            n_neighbors=n_neighbors,
            params=params,
            return_distance=return_distance,
            ctx=ctx,
            use_sandbox=use_sandbox,
            keep_tmp=keep_tmp,
        )
        response = _launch_plugin(spec, workspace, req_path, stderr_level=stderr_level)
        return _decode_result(
            workspace,
            request,
            response,
            return_distance,
            neighbor_name=neighbor_name,
        )


def _build_request(
    *,
    workspace: NNPluginWorkspace,
    data: np.ndarray,
    method: str,
    metric: str,
    n_neighbors: int,
    params: dict[str, Any],
    return_distance: bool,
    ctx: NNPluginContextInfo | None,
    use_sandbox: bool,
    keep_tmp: bool,
) -> tuple[NNPluginRequest, Path]:
    tmpdir = workspace.path
    result_path = tmpdir / "result.npz"
    response_path = tmpdir / "response.json"

    source_x = _find_source_data_path(ctx)
    sandbox = use_sandbox or source_x is None
    if sandbox:
        x_path = tmpdir / "x.npy"
        np.save(x_path, np.asarray(data, dtype=np.float32, order="C"))
        input_paths = NNPluginInputPaths(x_path=str(x_path))
    else:
        input_paths = NNPluginInputPaths(x_path=str(source_x))

    request = NNPluginRequest(
        protocol_version=NN_PLUGIN_PROTOCOL_VERSION,
        method=method,
        metric=metric,
        n_neighbors=int(n_neighbors),
        params=params,
        input=input_paths,
        options=NNPluginOptions(
            keep_temps=keep_tmp,
            use_sandbox_copies=sandbox,
        ),
        output=NNPluginOutputPaths(
            result_path=str(result_path),
            response_path=str(response_path),
        ),
    )
    req_path = tmpdir / "request.json"
    req_payload = request_to_dict(request)
    req_path.write_text(json.dumps(req_payload, ensure_ascii=False), encoding="utf-8")
    return request, req_path


def _launch_plugin(
    spec: NNPluginSpec,
    workspace: NNPluginWorkspace,
    req_path: Path,
    stderr_level: int,
) -> dict[str, Any]:
    cmd = list(spec.runner or _default_runner(spec))
    cmd += ["--method", workspace.method, "--request", str(req_path)]
    log.info("%s launching: %s", workspace.prefix, " ".join(cmd))
    env = _build_subprocess_env()
    _run_plugin_process(
        cmd,
        cwd=spec.plugin_dir,
        env=env,
        workspace=workspace,
        stderr_level=stderr_level,
    )
    return _load_response(req_path.parent / "response.json")


def _decode_result(
    workspace: NNPluginWorkspace,
    request: NNPluginRequest,
    response: dict[str, Any],
    return_distance: bool,
    neighbor_name: str | None = None,
) -> NearestNeighbors:
    if not response.get("ok", False):
        workspace.fail(f"plugin error: {response.get('message', 'unknown')}")

    npz_path = response["result_npz"]
    with np.load(npz_path, allow_pickle=False) as z:
        idx = z["idx"].astype(np.int32, copy=False)
        dist = None
        if "dist" in z.files:
            dist = z["dist"].astype(np.float32, copy=False)

    nn_name = neighbor_name or ""

    nn_info = NbrInfo(
        name=nn_name,
        n_nbrs=request.n_neighbors,
        metric=request.metric,
        exact=False,
        method=request.method,
        has_distances=dist is not None,
        idx_path=None,
        dist_path=None,
    )
    if return_distance and dist is None:
        workspace.fail("plugin omitted distances but return_distance=True")
    return NearestNeighbors(
        idx=idx, dist=dist if return_distance else None, info=nn_info
    )


def _build_subprocess_env() -> dict[str, str]:
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "DRNB_LOG_PLAIN": "1",
    }
    env.pop("VIRTUAL_ENV", None)
    env.pop("MPLBACKEND", None)
    return env


def _run_plugin_process(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    workspace: NNPluginWorkspace,
    stderr_level: int = logging.WARNING,
) -> None:
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    assert proc.stdout and proc.stderr
    stdout_logger = log.getChild(f"nn_plugin.{workspace.method}.stdout")
    stderr_logger = log.getChild(f"nn_plugin.{workspace.method}.stderr")
    stdout_thread = threading.Thread(
        target=_stream_pipe,
        args=(proc.stdout, stdout_logger, logging.INFO),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_stream_pipe,
        args=(proc.stderr, stderr_logger, stderr_level),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    code = proc.wait()
    stdout_thread.join()
    stderr_thread.join()
    if code != 0:
        workspace.fail(f"plugin exit {code}")


def _stream_pipe(pipe, logger, level: int) -> None:
    for line in pipe:
        logger.log(level, line.rstrip())
    pipe.close()


def _load_response(response_path: Path) -> dict[str, Any]:
    if not response_path.exists():
        raise NNPluginWorkspaceError(f"plugin response not written to {response_path}")
    try:
        return json.loads(response_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise NNPluginWorkspaceError(
            f"invalid plugin response at {response_path}: {exc}"
        ) from exc


_DATA_EXTS: tuple[str, ...] = (
    ".npy",
    ".npz",
    ".feather",
    ".parquet",
    ".pkl",
    ".pkl.gz",
    ".pkl.bz2",
    ".csv",
    ".csv.gz",
)


def _find_source_data_path(ctx: NNPluginContextInfo | None) -> Path | None:
    if ctx is None or ctx.drnb_home is None or ctx.dataset_name is None:
        return None
    data_dir = Path(ctx.drnb_home) / (ctx.data_sub_dir or "data")
    stems = [
        f"{ctx.dataset_name}-data",
        f"{ctx.dataset_name}_data",
        ctx.dataset_name,
    ]
    for stem in stems:
        for ext in _DATA_EXTS:
            candidate = data_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    return None


def _default_runner(spec: NNPluginSpec) -> list[str]:
    uv_var = os.environ.get("UV", "uv")
    uv_path = shutil.which(uv_var)
    if uv_path:
        return [uv_path, "run", "--color", "never", "--quiet", "drnb-nn-plugin-run.py"]
    raise NNPluginWorkspaceError(
        f"[nn-plugin:{spec.method}] uv executable '{uv_var}' not found in PATH; set UV to override"
    )
