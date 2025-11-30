# Environment Variables

These are the `DRNB_` environment variables a user can set to change runtime behavior. Anything not listed here is internal and used only for inter-process communication between the host and plugins.

| Variable | Default | What it does / when to set it |
| --- | --- | --- |
| `DRNB_HOME` | _required_ | Root directory for datasets and artifacts. Commands fail if unset. Example: `export DRNB_HOME=~/drnb-data`. |
| `DRNB_PLUGIN_KEEP_TMP` | `false` | Keep each plugin workspace instead of deleting it after a run. Set to `1` when debugging plugin outputs or inspecting the serialized request/inputs. Useful for debugging. |
| `DRNB_PLUGIN_SANDBOX_INPUTS` | `false` | Copy inputs into the plugin workspace instead of letting plugins read from data under `DRNB_HOME`. Possibly helpful if you are debugging an embedding method or need to copy everything somewhere else for closer examination. |
| `DRNB_PLUGINS_ROOT` | `plugins` folder inside the repo (or package) | Override the directory where plugins are discovered and launched. Point this at an external plugins checkout if you keep plugins separate from the main repo. |
| `DRNB_NN_PLUGINS_ROOT` | `nn-plugins` folder inside the repo (or package) | Override the directory where nearest-neighbor plugins are discovered and launched. |
| `DRNB_NN_PLUGIN_KEEP_TMP` | `false` | Keep each NN plugin workspace instead of deleting it after a run. Useful for debugging plugin inputs/outputs. |
| `DRNB_NN_FAISS` | `true` | By default, FAISS will be tried in most cases for default exact nearest neighbors, but in the very likely scenario it can't be installed, these failed attempts will fill the logs with error messages. If you would rather just not see this, set `DRNB_NN_FAISS=false` and these attempts will be skipped. Note that the default is `true`.|
| `DRNB_NN_PLUGIN_SANDBOX_INPUTS` | `false` | Copy NN plugin inputs into the workspace instead of reading data in place (zero-copy). Helpful when debugging or isolating runs. |
| `DRNB_LOG_PLAIN` | `false` | Disable rich/colored logging and emit plain text. No real reason to change this. This is used by the core of drnb in all subprocesses to avoid accidentally coloring already colored output. |

Boolean flags accept the usual truthy strings (`1`, `true`, `yes`, `on`, case-insensitive); use `0`, `false`, or leave unset to disable.
