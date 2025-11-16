# drnb-plugin-sdk

Minimal protocol definitions and optional Python helpers for drnb plugin runners.

- `drnb_plugin_sdk` (top-level package) exposes only the JSON dataclasses and
  request loader needed by every plugin. Copy this module verbatim if you need
  to target an older Python version or another language.
- `drnb_plugin_sdk.helpers` contains convenience utilities (neighbor IO, CLI
  runner, result writer). These depend on Python 3.12 and NumPy; use them only
  when you control the plugin runtime.

Keeping these concerns separate makes it possible to implement plugins in other
languages or Python versions without dragging along superfluous helpers.
