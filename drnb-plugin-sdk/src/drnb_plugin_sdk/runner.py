from __future__ import annotations

import argparse
import json
import sys
import traceback
from typing import Callable, Dict

from .protocol import PluginRequest, load_request


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
        print(
            json.dumps(
                {"ok": False, "message": f"unknown method {args.method}"},
            ),
            flush=True,
        )
        sys.exit(0)

    try:
        request = load_request(args.request)
        response = handler(request)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        response = {"ok": False, "message": str(exc)}

    print(json.dumps(response), flush=True)
