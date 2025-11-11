import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

from drnb.embed.base import Embedder
from drnb.embed.context import EmbedContext
from drnb.log import log  # rich logger
from drnb.plugins.protocol import (
    PROTOCOL_VERSION,
    PluginInputPaths,
    PluginNeighbors,
    PluginOptions,
    PluginOutputPaths,
    PluginRequest,
    context_to_payload,
    env_flag,
    request_to_dict,
    sanitize_params,
)
from drnb.plugins.registry import get_registry, plugins_enabled
from drnb.types import EmbedResult


@dataclass
class ExternalEmbedder(Embedder):
    """
    Out-of-process embedder for conflict-heavy methods. Returns the same
    result shape as in-process embedders: {"coords": ..., "snapshots": {...}}.
    """

    # Make 'method' kw-only with a default to avoid dataclass ordering issues.
    method: str = field(default="", kw_only=True)
    # Accept both spellings; we'll resolve at runtime.
    use_precomputed_knn: bool | None = None
    use_precomputed_neighbors: bool | None = None
    drnb_init: str | None = None
    snapshots: list[int] | None = None

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
            snapshots = sorted({int(s) for s in (self.snapshots or [])})
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
                    snapshots=snapshots,
                    keep_temps=keep_tmp,
                    use_precomputed_knn=use_knn,
                ),
                output=PluginOutputPaths(result_path=str(result_path)),
            )
            req_path = tmpdir / "request.json"
            req_payload = request_to_dict(request)
            req_path.write_text(
                json.dumps(req_payload, ensure_ascii=False), encoding="utf-8"
            )

            # Build command. Default: current python, unbuffered, run the runner script.
            cmd = spec.runner or [sys.executable, "-u", "drnb-plugin-run.py"]
            cmd = list(cmd) + ["--method", self.method, "--request", str(req_path)]

            log.info(f"[external:{self.method}] launching: {' '.join(cmd)}")

            # Stream plugin logs from stderr, keep stdout for the final JSON line.
            env = {
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "DRNB_LOG_PLAIN": "1",
            }
            proc = subprocess.Popen(  # noqa: S603
                cmd,
                cwd=spec.plugin_dir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            assert proc.stdout and proc.stderr
            try:
                for line in proc.stderr:
                    log.info(line.rstrip())  # live logs into notebook
                out = proc.stdout.read()
                code = proc.wait()
            finally:
                try:
                    proc.kill()
                except Exception:
                    pass

            if code != 0:
                self._fail(f"plugin exit {code}")

            try:
                resp = json.loads(out.strip())
            except Exception as e:  # noqa: BLE001
                self._fail(f"bad JSON from plugin: {e}")

            if not resp.get("ok", False):
                self._fail(f"plugin error: {resp.get('message', 'unknown')}")

            npz_hint = resp.get("result_npz") or request.output.result_path
            npz_path = Path(npz_hint).resolve()
            if not _path_within(npz_path, tmpdir):
                self._fail("plugin wrote results outside of workspace")

            with np.load(npz_path, allow_pickle=False) as z:
                coords = z["coords"].astype(np.float32, copy=False)
                snaps: dict[int, np.ndarray] = {}
                for k in z.files:
                    if k.startswith("snap_"):
                        try:
                            it = int(k.split("_")[1])
                            snaps[it] = z[k].astype(np.float32, copy=False)
                        except Exception:
                            pass

            result: dict[str, Any] = {"coords": coords}
            if snaps:
                result["snapshots"] = snaps
            return result

        finally:
            if keep_tmp:
                log.info(f"[external:{self.method}] kept plugin workspace at {tmpdir}")
            else:
                shutil.rmtree(tmpdir, ignore_errors=True)

    def embed(self, x: np.ndarray, ctx: EmbedContext | None = None) -> EmbedResult:
        return self.embed_impl(x, self.params, ctx)

    def _fail(self, message: str) -> NoReturn:
        raise RuntimeError(f"[external:{self.method}] {message}")


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
