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
)

from drnb.embed.base import Embedder
from drnb.embed.context import EmbedContext
from drnb.log import log
from drnb.plugins.protocol import (
    context_to_payload,
    env_flag,
    request_to_dict,
    sanitize_params,
)
from drnb.plugins.registry import PluginSpec, get_registry, plugins_enabled
from drnb.types import EmbedResult


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
    drnb_init: str | None = None

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
            self._fail("plugins disabled via DRNB_PLUGINS")

        spec = get_registry().lookup(self.method)
        if spec is None or not spec.plugin_dir.exists():
            self._fail("plugin not found")

        params = dict(params or {})
        safe_params = sanitize_params(params)
        keep_tmp = env_flag("DRNB_PLUGIN_KEEP_TMP", False)
        tmpdir = Path(tempfile.mkdtemp(prefix=f"drnb-{self.method}-"))
        self._workspace_dir = tmpdir
        self._cleanup_workspace = not keep_tmp

        try:
            x_path = tmpdir / "x.npy"
            np.save(x_path, np.asarray(x, dtype=np.float32, order="C"))

            # Pass precomputed KNN if ctx allows it; failure is soft (plugin can recompute)
            idx_path = dist_path = None
            if use_knn and ctx is not None:
                try:
                    from drnb.embed.context import get_neighbors_with_ctx

                    metric = (
                        params.get("metric") or params.get("distance") or "euclidean"
                    )
                    n_neighbors = int(params.get("n_neighbors", 15))
                    pre = get_neighbors_with_ctx(x, metric, n_neighbors, ctx=ctx)
                    if pre is not None and getattr(pre, "idx", None) is not None:
                        idx_path = tmpdir / "knn_idx.npy"
                        np.save(idx_path, pre.idx.astype(np.int32, copy=False))
                        if getattr(pre, "dist", None) is not None:
                            dist_path = tmpdir / "knn_dist.npy"
                            np.save(dist_path, pre.dist.astype(np.float32, copy=False))
                except Exception as e:  # noqa: BLE001
                    log.warning(
                        f"[external:{self.method}] KNN passthrough failed; plugin may compute: {e}"
                    )

            result_path = tmpdir / "result.npz"
            response_path = tmpdir / "response.json"
            input_paths = PluginInputPaths(
                x_path=str(x_path),
                neighbors=PluginNeighbors(
                    idx_path=str(idx_path) if idx_path else None,
                    dist_path=str(dist_path) if dist_path else None,
                ),
            )

            if self.drnb_init is not None:
                init_path = tmpdir / "init.npy"
                np.save(
                    init_path,
                    np.asarray(self.drnb_init, dtype=np.float32, order="C"),
                )
                input_paths.init_path = str(init_path)

            request = PluginRequest(
                protocol_version=PROTOCOL_VERSION,
                method=self.method,
                params=safe_params,
                context=context_to_payload(ctx),
                input=input_paths,
                options=PluginOptions(
                    keep_temps=keep_tmp,
                    use_precomputed_knn=use_knn,
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

            log.info(f"[external:{self.method}] launching: {' '.join(cmd)}")

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
                self._fail(f"plugin exit {code}")

            resp = _load_response(response_path)

            if not resp.get("ok", False):
                self._fail(f"plugin error: {resp.get('message', 'unknown')}")

            npz_hint = resp.get("result_npz") or request.output.result_path
            npz_path = Path(npz_hint).resolve()
            if not _path_within(npz_path, tmpdir):
                self._fail("plugin wrote results outside of workspace")

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

        finally:
            cleanup = getattr(self, "_cleanup_workspace", True)
            if cleanup:
                shutil.rmtree(tmpdir, ignore_errors=True)
            else:
                log.info(f"[external:{self.method}] kept plugin workspace at {tmpdir}")
            self._workspace_dir = None
            self._cleanup_workspace = True

    def embed(self, x: np.ndarray, ctx: EmbedContext | None = None) -> EmbedResult:
        return self.embed_impl(x, self.params, ctx)

    def _fail(self, message: str) -> NoReturn:
        workspace = getattr(self, "_workspace_dir", None)
        if workspace is not None:
            log.error(
                f"[external:{self.method}] failure occurred; workspace retained at {workspace}"
            )
            message = f"{message} (workspace: {workspace})"
            self._cleanup_workspace = False
        raise RuntimeError(f"[external:{self.method}] {message}")


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
