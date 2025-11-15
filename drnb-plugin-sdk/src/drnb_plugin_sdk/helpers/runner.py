from __future__ import annotations

import argparse
import sys
import traceback
from typing import Callable, Dict

from drnb_plugin_sdk.protocol import PluginRequest, load_request

from .results import write_response_json


def run_plugin(
    handlers: Dict[str, Callable[[PluginRequest], dict]],
    *,
    description: str | None = None,
) -> None:
    """Generic CLI entrypoint for plugin runners."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--method", required=True)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()

    handler = handlers.get(args.method)
    if handler is None:
        print(f"Unknown method {args.method}", file=sys.stderr, flush=True)
        sys.exit(1)

    try:
        request = load_request(args.request)
        response = handler(request)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        response = {"ok": False, "message": str(exc)}

    response_path = request.output.response_path
    if not response_path:
        raise RuntimeError("Plugin request missing output.response_path")
    write_response_json(response_path, response)
