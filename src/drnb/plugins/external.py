from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
from drnb_plugin_sdk import (
    PROTOCOL_VERSION,
    PluginInputPaths,
    PluginNeighbors,
    PluginOptions,
    PluginOutputPaths,
    PluginRequest,
    PluginSourcePaths,
    env_flag,
    request_to_dict,
    sanitize_params,
)

from drnb.embed.base import Embedder
from drnb.embed.context import EmbedContext
from drnb.log import log
from drnb.neighbors.store import find_candidate_neighbors_info
from drnb.plugins.protocol import (
    context_to_payload,
)
from drnb.plugins.registry import PluginSpec, get_registry, plugins_enabled
from drnb.types import EmbedResult


@dataclass
class PluginWorkspace:
    """Workspace for a plugin run.

    Args:
        path: The path to the workspace.
        remove_on_exit: Whether to remove the workspace on exit.
        method: The method name.

    In practice, `remove_on_exit` is set by the value of the `DRNB_PLUGIN_KEEP_TMP`
    environment variable.
    """

    path: Path
    remove_on_exit: bool
    method: str

    @property
    def prefix(self) -> str:
        return f"[external:{self.method}]"

    def fail(self, message: str) -> NoReturn:
        """Raise with context and retain the workspace even if remove_on_exit is
        True."""
        self.remove_on_exit = False
        log.error(
            "%s failure occurred; workspace retained at %s",
            self.prefix,
            self.path,
        )
        raise RuntimeError(f"{self.prefix} {message} (workspace: {self.path})")

    def __enter__(self) -> PluginWorkspace:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Preserve workspace on unexpected errors unless remove_on_exit explicitly
        # asked to remove.
        if exc_type is not None and self.remove_on_exit:
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
class ExternalEmbedder(Embedder):
    """
    Out-of-process embedder for conflict-heavy methods. Returns the same
    result shape as in-process embedders: {"coords": ..., }.
    """

    # Make 'method' kw-only with a default to avoid dataclass ordering issues.
    method: str = field(default="", kw_only=True)
    # Accept both spellings; we'll resolve at runtime.
    use_precomputed_knn: bool | None = None
    use_precomputed_neighbors: bool | None = None
    drnb_init: str | Path | None = None
    use_sandbox_copies: bool | None = None

    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> EmbedResult:
        # Resolve precomputed-knn preference
        use_knn = (
            self.use_precomputed_knn
            if self.use_precomputed_knn is not None
            else (
                self.use_precomputed_neighbors
                if self.use_precomputed_neighbors is not None
                else True
            )
        )

        if not plugins_enabled():
            raise RuntimeError(
                f"[external:{self.method}] plugins disabled via DRNB_PLUGINS"
            )

        spec = get_registry().lookup(self.method)
        if spec is None or not spec.plugin_dir.exists():
            raise RuntimeError(f"[external:{self.method}] plugin not found")

        params = dict(params or {})
        safe_params = sanitize_params(params)
        keep_tmp = env_flag("DRNB_PLUGIN_KEEP_TMP", False)
        sandbox_env = env_flag("DRNB_PLUGIN_SANDBOX_INPUTS", False)
        use_sandbox = (
            sandbox_env if self.use_sandbox_copies is None else self.use_sandbox_copies
        )
        workspace = PluginWorkspace(
            path=Path(tempfile.mkdtemp(prefix=f"drnb-{self.method}-")),
            remove_on_exit=not keep_tmp,
            method=self.method,
        )
        tmpdir = workspace.path

        with workspace:
            result_path = tmpdir / "result.npz"
            response_path = tmpdir / "response.json"

            source_x = _find_source_data_path(ctx)
            init_source = _init_source_path(self.drnb_init)
            source_neighbors = _find_source_neighbors(ctx, params) if use_knn else None

            if self.drnb_init is not None and init_source is None:
                workspace.fail(f"init path not found: {self.drnb_init}")

            source_paths = None
            if ctx is not None and ctx.drnb_home is not None:
                source_paths = PluginSourcePaths(
                    drnb_home=Path(ctx.drnb_home),
                    dataset=ctx.dataset_name,
                    data_sub_dir=ctx.data_sub_dir,
                    nn_sub_dir=ctx.nn_sub_dir,
                    triplet_sub_dir=ctx.triplet_sub_dir,
                    x_path=str(source_x) if source_x else None,
                    init_path=str(init_source) if init_source else None,
                    neighbors=source_neighbors or PluginNeighbors(),
                )

            input_paths = PluginInputPaths(
                x_path="",
                neighbors=PluginNeighbors(),
                source_paths=source_paths,
            )

            if use_sandbox:
                x_path = tmpdir / "x.npy"
                np.save(x_path, np.asarray(x, dtype=np.float32, order="C"))
                input_paths.x_path = str(x_path)
                input_paths.neighbors = _prepare_neighbor_paths(
                    tmpdir,
                    use_knn=use_knn,
                    source_neighbors=source_neighbors,
                    params=params,
                    x=x,
                    ctx=ctx,
                    method=self.method,
                )
                init_path = _prepare_init_path(tmpdir, init_source)
                input_paths.init_path = str(init_path) if init_path else None
            else:
                missing_inputs: list[str] = []
                if source_x is None:
                    missing_inputs.append("x")
                if missing_inputs:
                    missing = ", ".join(missing_inputs)
                    workspace.fail(f"missing source inputs for zero-copy: {missing}")
                input_paths.x_path = str(source_x)
                input_paths.neighbors = source_neighbors or PluginNeighbors()
                input_paths.init_path = str(init_source) if init_source else None

            request = PluginRequest(
                protocol_version=PROTOCOL_VERSION,
                method=self.method,
                params=safe_params,
                context=context_to_payload(ctx),
                input=input_paths,
                options=PluginOptions(
                    keep_temps=keep_tmp,
                    use_precomputed_knn=use_knn,
                    use_sandbox_copies=use_sandbox,
                ),
                output=PluginOutputPaths(
                    result_path=str(result_path), response_path=str(response_path)
                ),
            )
            req_path = tmpdir / "request.json"
            req_payload = request_to_dict(request)
            req_path.write_text(
                json.dumps(req_payload, ensure_ascii=False), encoding="utf-8"
            )

            cmd = list(spec.runner or _default_runner(spec))
            cmd += ["--method", self.method, "--request", str(req_path)]

            log.info(f"{workspace.prefix} launching: {' '.join(cmd)}")

            # Stream plugin logs from stdout/stderr, let response JSON be written to disk.
            env = {
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "DRNB_LOG_PLAIN": "1",
            }
            env.pop("VIRTUAL_ENV", None)
            # pymde couldn't run from a notebook in Cursor (probably a VS Code issue?)
            # because MPLBACKEND was set to 'module://matplotlib_inline.backend_inline'
            # and matplotlib doesn't know how to handle this without installing the
            # `matplotlib-inline` package. As we don't want to use matplotlib
            # functionality from pymde, just unset this environment variable when the
            # subprocess is launched. This should be safe with other plugins as we are
            # not looking for matplotlib functionality from them.
            env.pop("MPLBACKEND", None)
            proc = subprocess.Popen(  # noqa: S603
                cmd,
                cwd=spec.plugin_dir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            assert proc.stdout and proc.stderr
            stdout_logger = log.getChild(f"external.{self.method}.stdout")
            stderr_logger = log.getChild(f"external.{self.method}.stderr")
            stdout_thread = threading.Thread(
                target=_stream_pipe,
                args=(proc.stdout, stdout_logger, logging.INFO),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=_stream_pipe,
                args=(proc.stderr, stderr_logger, logging.INFO),
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()
            code = proc.wait()
            stdout_thread.join()
            stderr_thread.join()

            if code != 0:
                workspace.fail(f"plugin exit {code}")

            resp = _load_response(response_path)

            if not resp.get("ok", False):
                workspace.fail(f"plugin error: {resp.get('message', 'unknown')}")

            npz_hint = resp.get("result_npz") or request.output.result_path
            npz_path = Path(npz_hint).resolve()
            if not _path_within(npz_path, tmpdir):
                workspace.fail("plugin wrote results outside of workspace")

            with np.load(npz_path, allow_pickle=False) as z:
                coords = z["coords"].astype(np.float32, copy=False)
                snaps: dict[str, np.ndarray] = {}
                for k in z.files:
                    if k.startswith("snap_"):
                        try:
                            it = int(k.split("_")[1])
                            snaps[f"it_{it}"] = z[k].astype(np.float32, copy=False)
                        except Exception:
                            pass

            result: dict[str, Any] = {"coords": coords}
            if snaps:
                result["snapshots"] = snaps
            return result

    def embed(self, x: np.ndarray, ctx: EmbedContext | None = None) -> EmbedResult:
        return self.embed_impl(x, self.params, ctx)


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _default_runner(spec: PluginSpec) -> list[str]:
    uv_var = os.environ.get("UV", "uv")
    uv_path = shutil.which(uv_var)
    if uv_path:
        return [uv_path, "run", "--color", "never", "--quiet", "drnb-plugin-run.py"]

    plugin_python = _find_plugin_python(spec.plugin_dir)
    if plugin_python:
        log.warning(
            "[external:%s] uv executable '%s' not found; using plugin-local interpreter %s",
            spec.method,
            uv_var,
            plugin_python,
        )
        return [plugin_python, "-u", "drnb-plugin-run.py"]

    log.warning(
        "[external:%s] uv executable '%s' not found and plugin .venv is missing; using host interpreter",
        spec.method,
        uv_var,
    )
    return [sys.executable, "-u", "drnb-plugin-run.py"]


def _find_plugin_python(plugin_dir: Path) -> str | None:
    candidates = [
        plugin_dir / ".venv" / "bin" / "python",
        plugin_dir / ".venv" / "Scripts" / "python.exe",
        plugin_dir / ".venv" / "Scripts" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _stream_pipe(pipe, logger, level: int) -> None:
    try:
        for line in pipe:
            logger.log(level, line.rstrip())
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _load_response(response_path: Path | str) -> dict[str, Any]:
    path = Path(response_path)
    if not path.exists():
        raise RuntimeError(f"plugin response not written to {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"invalid plugin response at {path}: {exc}") from exc


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


def _find_source_data_path(ctx: EmbedContext | None) -> Path | None:
    if ctx is None or ctx.drnb_home is None:
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


def _init_source_path(drnb_init: str | Path | None) -> Path | None:
    if drnb_init is None:
        return None
    if isinstance(drnb_init, (str, Path)):
        path = Path(drnb_init)
        if path.exists():
            return path
    return None


def _find_source_neighbors(
    ctx: EmbedContext | None, params: dict[str, Any]
) -> PluginNeighbors | None:
    if ctx is None or ctx.drnb_home is None:
        return None
    metric = params.get("metric") or params.get("distance") or "euclidean"
    names = []
    try:
        embed_nn = ctx.embed_nn_name
    except Exception:
        embed_nn = None
    if embed_nn:
        names.append(embed_nn)
    if ctx.dataset_name not in names:
        names.append(ctx.dataset_name)
    for name in names:
        # Note that we purposely set `n_neighbors=None` here so that the result with
        # the maximum number of neighbors is returned. This is because the plugin
        # embedder may ask for a larger number of neighbors based on any `n_neighbors`
        # parameter and we have no way to know that here.
        info = find_candidate_neighbors_info(
            name=name,
            drnb_home=ctx.drnb_home,
            sub_dir=ctx.nn_sub_dir or "nn",
            n_neighbors=None,
            metric=metric,
            return_distance=True,
            verbose=False,
        )
        if info is not None:
            return PluginNeighbors(
                idx_path=str(info.idx_path) if info.idx_path else None,
                dist_path=str(info.dist_path) if info.dist_path else None,
            )
    return None


def _prepare_neighbor_paths(
    tmpdir: Path,
    *,
    use_knn: bool,
    source_neighbors: PluginNeighbors | None,
    params: dict[str, Any],
    x: np.ndarray,
    ctx: EmbedContext | None,
    method: str,
) -> PluginNeighbors:
    if not use_knn:
        return PluginNeighbors()
    if source_neighbors and source_neighbors.idx_path:
        return _copy_neighbor_files(tmpdir, source_neighbors)
    try:
        from drnb.embed.context import get_neighbors_with_ctx

        metric = params.get("metric") or params.get("distance") or "euclidean"
        n_neighbors = int(params.get("n_neighbors", 15))
        pre = get_neighbors_with_ctx(x, metric, n_neighbors, ctx=ctx)
        if pre is None or getattr(pre, "idx", None) is None:
            return PluginNeighbors()
        return _write_neighbor_arrays(tmpdir, pre.idx, getattr(pre, "dist", None))
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[external:%s] KNN passthrough failed; plugin may compute: %s", method, exc
        )
        return PluginNeighbors()


def _prepare_init_path(tmpdir: Path, init_source: Path | None) -> Path | None:
    """Copy the provided init file into the workspace."""
    if init_source is None:
        return None
    target = tmpdir / Path(init_source).name
    shutil.copy(init_source, target)
    return target


def _copy_neighbor_files(tmpdir: Path, neighbors: PluginNeighbors) -> PluginNeighbors:
    idx_path = dist_path = None
    if neighbors.idx_path:
        src = Path(neighbors.idx_path)
        if src.exists():
            idx_path = tmpdir / src.name
            shutil.copy(src, idx_path)
    if neighbors.dist_path:
        src = Path(neighbors.dist_path)
        if src.exists():
            dist_path = tmpdir / src.name
            shutil.copy(src, dist_path)
    return PluginNeighbors(
        idx_path=str(idx_path) if idx_path else None,
        dist_path=str(dist_path) if dist_path else None,
    )


def _write_neighbor_arrays(
    tmpdir: Path, idx: np.ndarray, dist: np.ndarray | None
) -> PluginNeighbors:
    idx_path = tmpdir / "knn_idx.npy"
    np.save(idx_path, np.asarray(idx, dtype=np.int32, order="C"))
    dist_path = None
    if dist is not None:
        dist_path = tmpdir / "knn_dist.npy"
        np.save(dist_path, np.asarray(dist, dtype=np.float32, order="C"))
    return PluginNeighbors(
        idx_path=str(idx_path),
        dist_path=str(dist_path) if dist_path else None,
    )
