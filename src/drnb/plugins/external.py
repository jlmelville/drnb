import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from drnb.embed.base import Embedder
from drnb.embed.context import EmbedContext
from drnb.log import log  # rich logger
from drnb.plugins.registry import get_registry, plugins_enabled
from drnb.types import EmbedResult


def _placeholder_coords(n: int, seed: int = 42) -> np.ndarray:
    # reasonably compact spiral to avoid plot sizing surprises
    k = np.arange(n, dtype=np.float32)
    r = np.sqrt(k / max(n - 1, 1))
    phi = (np.pi * (3 - np.sqrt(5))) * k
    return np.column_stack([r * np.cos(phi), r * np.sin(phi)]).astype(np.float32)


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
    on_unavailable: str = "placeholder"  # "placeholder" | "error"

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

        if not (
            plugins_enabled()
            and (spec := get_registry().lookup(self.method))
            and spec.plugin_dir.exists()
        ):
            return self._unavailable(x, f"plugin not found or plugins disabled")

        with tempfile.TemporaryDirectory(prefix=f"drnb-{self.method}-") as td:
            tdir = Path(td)
            x_path = tdir / "x.npy"
            np.save(x_path, np.asarray(x, dtype=np.float32, order="C"))

            # Pass precomputed KNN if ctx allows it; failure is soft (plugin can recompute)
            idx_path = dist_path = None
            if use_knn and ctx is not None:
                try:
                    from drnb.embed.context import get_neighbors_with_ctx

                    metric = (
                        self.params.get("metric")
                        or self.params.get("distance")
                        or "euclidean"
                    )
                    n_neighbors = int(self.params.get("n_neighbors", 15))
                    pre = get_neighbors_with_ctx(x, metric, n_neighbors, ctx=ctx)
                    if pre is not None and getattr(pre, "idx", None) is not None:
                        idx_path = tdir / "knn_idx.npy"
                        np.save(idx_path, pre.idx.astype(np.int32, copy=False))
                        if getattr(pre, "dist", None) is not None:
                            dist_path = tdir / "knn_dist.npy"
                            np.save(dist_path, pre.dist.astype(np.float32, copy=False))
                except Exception as e:  # noqa: BLE001
                    log.warning(
                        f"[external:{self.method}] KNN passthrough failed; plugin may compute: {e}"
                    )

            req = {
                "protocol": 1,
                "method": self.method,
                "params": dict(self.params),
                "options": {
                    "snapshots": sorted(set(self.snapshots or [])),
                },
                "input": {
                    "x_path": str(x_path),
                    "neighbors": {
                        "idx_path": str(idx_path) if idx_path else None,
                        "dist_path": str(dist_path) if dist_path else None,
                    },
                },
                "context": {
                    "dataset_name": getattr(ctx, "dataset_name", ""),
                    "embed_method_label": getattr(
                        ctx, "embed_method_label", self.method
                    ),
                },
            }
            req_path = tdir / "request.json"
            req_path.write_text(json.dumps(req), encoding="utf-8")

            # Build command. Default: current python, unbuffered, run the runner script.
            cmd = spec.runner or [sys.executable, "-u", "drnb-plugin-run.py"]
            cmd = list(cmd) + ["--method", self.method, "--request", str(req_path)]

            log.info(f"[external:{self.method}] launching: {' '.join(cmd)}")

            # Stream plugin logs from stderr, keep stdout for the final JSON line.
            proc = subprocess.Popen(  # noqa: S603
                cmd,
                cwd=spec.plugin_dir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
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
                msg = f"plugin exit {code}"
                log.warning(f"[external:{self.method}] {msg}")
                return self._unavailable(x, msg)

            try:
                resp = json.loads(out.strip())
            except Exception as e:  # noqa: BLE001
                return self._unavailable(x, f"bad JSON from plugin: {e}")

            if not resp.get("ok", False):
                return self._unavailable(
                    x, f"plugin error: {resp.get('message', 'unknown')}"
                )

            npz_path = Path(resp["result_npz"])
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

    def embed(self, x: np.ndarray, ctx: EmbedContext | None = None) -> EmbedResult:
        return self.embed_impl(x, self.params, ctx)

    def _unavailable(self, x: np.ndarray, reason: str) -> dict[str, Any]:
        if self.on_unavailable == "error":
            raise RuntimeError(reason)
        log.warning(f"[external:{self.method}] unavailable -> placeholder ({reason})")
        return {
            "coords": _placeholder_coords(x.shape[0]),
            "info": {"unavailable": True, "reason": reason},
        }
