# drnb-plugin-sdk

Shared helpers used by the `drnb` core and its external plugin runners. This package
exposes the IPC protocol dataclasses, CLI runner wrapper, neighbor file utilities,
and result-writing helpers so plugins can stay lightweight without importing the full
`drnb` dependency stack.

Install in editable mode while working inside the main repository:

```
uv pip install -e drnb-plugin-sdk
```

After installation, both the host environment and every plugin virtual environment can
import `drnb_plugin_sdk`.

## Neighbor helpers

Plugins should call `drnb_plugin_sdk.neighbors.load_neighbors(request)` to load the
optional `knn_idx.npy` / `knn_dist.npy` files referenced inside the request payload.
It is up to each plugin to decide how many neighbors it needs (by inspecting user
parameters, defaults, or method-specific heuristics) and then slice or supplement the
loaded arrays accordingly. If no neighbor files exist for the requested dataset, the
SDK helper returns `(None, None)` so plugins can fall back to their own neighbor logic.
