# NN Plugin Runner Protocol

This document describes the contract between `drnb` (the host) and every nearest-neighbor (NN) plugin runner. The protocol mirrors the embedder plugin flow: the host writes a JSON request plus serialized inputs into a temporary workspace, the plugin writes its outputs and a final JSON response, and the host loads the result. Zero-copy is the default; sandbox copies are opt-in.

## Request layout

`ExternalNeighbors` writes a JSON request file alongside the serialized inputs in a temporary workspace. The payload mirrors `drnb_nn_plugin_sdk.protocol.NNPluginRequest` and contains:

- `protocol_version`: currently `1`. Plugins must validate this integer before doing any work.
- `method`: registry key for the NN method (e.g., `annoy`). No `-plugin` suffix in steady state.
- `metric`: distance metric string. Must match the core/in-process naming (e.g., `euclidean`, `cosine`).
- `n_neighbors`: maximum number of neighbors the plugin should compute. The host may later slice down when writing cache files.
- `params`: method parameters (JSON primitives only) using the same names/defaults as the in-process implementation.
- `input.x_path`: required feature matrix path. Absolute paths are provided. No other inputs are required for NN plugins.
- `options`: flags such as `use_sandbox_copies` (default: false) and `keep_temps`; `log_path` is reserved.
- `output.result_path`: where the plugin must write its `.npz` result (containing `idx` and `dist`).
- `output.response_path`: where the plugin must write the final JSON response.

## Plugin responsibilities

1. Read the request JSON, either manually or via `drnb_nn_plugin_sdk.helpers.runner.run_nn_plugin`, which loads the request and dispatches to the handler.
2. Compute nearest neighbors using only the inputs and parameters provided. Do not import `drnb` core modules.
3. Write a compressed `.npz` to `req.output.result_path` containing:
   - `idx`: neighbor indices as int32
   - `dist`: neighbor distances as float32 (always present for now)
4. Write the final JSON response to `req.output.response_path`, e.g. `{"ok": true, "result_npz": "<path>"}` or `{"ok": false, "message": "..."}`. `helpers.results.save_neighbors_npz` returns a ready response dict; `helpers.results.write_response_json` persists it.
5. Print diagnostics freely to stdout/stderr; the host streams both pipes with prefixes so users can trace plugin output.

## Host expectations

The host will launch each plugin with `uv run --quiet --color never drnb-nn-plugin-run.py --method <name> --request <path>` from the plugin directory, stripping `VIRTUAL_ENV` so the plugin's own `.venv` is used. After the process exits, the host reads `output.response_path` and loads the JSON payload. If the file is missing or invalid, the run is treated as a fatal error. The host also checks that the result `.npz` stays inside the workspace and then loads `idx`/`dist` to build a `NearestNeighbors` object before writing canonical cache files.

## Sample request (zero-copy)

  {
    "protocol": 1,
    "method": "annoy",
    "metric": "euclidean",
    "n_neighbors": 16,
    "params": {
      "n_trees": 50,
      "search_k": -1,
      "random_state": 42,
      "n_jobs": -1
    },
    "input": {
      "x_path": "/home/data/datasets/data/s1k-data.npy"
    },
    "options": {
      "keep_temps": false,
      "log_path": null,
      "use_sandbox_copies": false
    },
    "output": {
      "result_path": "/tmp/drnb-annoy-XXXX/result.npz",
      "response_path": "/tmp/drnb-annoy-XXXX/response.json"
    }
  }

## Response examples

- Success: `{"ok": true, "result_npz": "/tmp/drnb-annoy-XXXX/result.npz"}`
- Error: `{"ok": false, "message": "annoy build failed"}`
