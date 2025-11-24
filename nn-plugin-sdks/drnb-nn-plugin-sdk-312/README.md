# drnb-nn-plugin-sdk

Protocol definitions and lightweight helpers for nearest-neighbor (NN) plugins used by `drnb`. This SDK targets Python 3.12 and is intentionally small so NN plugins can stay isolated from the core `drnb` dependency graph.

- `drnb_nn_plugin_sdk` exposes the IPC protocol dataclasses plus JSON load/save helpers. Use this to read the host-generated request in a plugin and write back the response.
- `drnb_nn_plugin_sdk.helpers` contains convenience utilities (runner CLI, result writer, basic logging/param summarization). These depend only on NumPy and the standard library.

The SDK mirrors the embedder plugin SDK in spirit: zero-copy by default, workspace-sandbox opt-in, and a simple request/response contract written to disk. It does **not** pull in any neighbor-compute libraries; plugins bring their own dependencies (e.g., Annoy, hnswlib, FAISS).
