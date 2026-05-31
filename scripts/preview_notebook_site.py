"""Render, optimize, and serve the Quarto notebook site locally."""

from __future__ import annotations

import argparse
import functools
import http.server
import socket
import socketserver
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4202


class ReusableTcpServer(socketserver.TCPServer):
    allow_reuse_address = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Host interface to bind the preview server to; default: {DEFAULT_HOST}.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Preferred preview server port; default: {DEFAULT_PORT}.",
    )
    parser.add_argument(
        "--strict-port",
        action="store_true",
        help="Fail if --port is unavailable instead of trying later ports.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Serve the existing notebooks/_site output without running Quarto.",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip the rendered PNG optimization step.",
    )
    parser.add_argument(
        "--max-edge",
        type=int,
        default=3200,
        help="Maximum PNG width or height passed to scripts/optimize_site_images.py.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the preview URL in the default browser.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch notebook-site inputs and re-render when they change.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Seconds between filesystem checks when --watch is enabled.",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def port_is_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
        return True


def choose_port(host: str, preferred_port: int, *, strict: bool) -> int:
    if strict:
        if not port_is_available(host, preferred_port):
            raise SystemExit(f"{host}:{preferred_port} is already in use")
        return preferred_port

    for port in range(preferred_port, preferred_port + 100):
        if port_is_available(host, port):
            return port

    raise SystemExit(
        f"No available ports found from {preferred_port} to {preferred_port + 99}"
    )


def optimize_site(site_dir: Path, *, max_edge: int) -> None:
    run(
        [
            "uv",
            "run",
            "--with",
            "pillow==11.1.0",
            "--no-project",
            "python",
            "scripts/optimize_site_images.py",
            str(site_dir.relative_to(REPO_ROOT)),
            "--max-edge",
            str(max_edge),
        ]
    )


def build_site(args: argparse.Namespace, site_dir: Path) -> None:
    if not args.no_render:
        run(["quarto", "render", "notebooks"])

    if not args.no_optimize:
        optimize_site(site_dir, max_edge=args.max_edge)

    if not (site_dir / "index.html").is_file():
        raise SystemExit(f"{site_dir} does not look like a rendered site")


def iter_watch_paths() -> list[Path]:
    notebooks = REPO_ROOT / "notebooks"
    paths = [
        notebooks / "_quarto.yml",
        notebooks / "_open-image.html",
        notebooks / "index.qmd",
        REPO_ROOT / "scripts" / "optimize_site_images.py",
    ]
    paths.extend(sorted((notebooks / "articles").glob("*.ipynb")))
    return [path for path in paths if path.exists()]


def snapshot_watch_paths() -> dict[Path, int]:
    return {path: path.stat().st_mtime_ns for path in iter_watch_paths()}


def watch_and_rebuild(args: argparse.Namespace, site_dir: Path) -> None:
    previous = snapshot_watch_paths()
    print("Watching notebook-site inputs for changes.", flush=True)

    while True:
        time.sleep(args.poll_interval)
        current = snapshot_watch_paths()
        if current == previous:
            continue

        changed = sorted(
            path.relative_to(REPO_ROOT)
            for path in set(previous) | set(current)
            if previous.get(path) != current.get(path)
        )
        print(
            "Detected change in "
            + ", ".join(str(path) for path in changed)
            + "; rebuilding.",
            flush=True,
        )

        try:
            build_site(args, site_dir)
        except subprocess.CalledProcessError as exc:
            print(
                f"Build failed with exit code {exc.returncode}; still watching.",
                file=sys.stderr,
            )
        else:
            print("Rebuild complete. Refresh the browser to view changes.", flush=True)
            previous = current


def main() -> None:
    args = parse_args()
    site_dir = REPO_ROOT / "notebooks" / "_site"

    if args.watch and args.no_render:
        raise SystemExit("--watch requires rendering; remove --no-render")

    build_site(args, site_dir)

    port = choose_port(args.host, args.port, strict=args.strict_port)
    url = f"http://{args.host}:{port}/"
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=site_dir
    )

    print(f"Serving {site_dir.relative_to(REPO_ROOT)} at {url}", flush=True)
    print("Press Ctrl-C to stop.", flush=True)

    if args.open:
        webbrowser.open(url)

    if args.watch:
        watcher = threading.Thread(
            target=watch_and_rebuild,
            args=(args, site_dir),
            daemon=True,
        )
        watcher.start()

    try:
        with ReusableTcpServer((args.host, port), handler) as server:
            server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped preview server.", file=sys.stderr)


if __name__ == "__main__":
    main()
