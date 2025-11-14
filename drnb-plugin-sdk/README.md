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
