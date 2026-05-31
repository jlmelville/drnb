"""Optimize rendered Quarto site PNG images in place."""

from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def iter_pngs(root: Path) -> Iterable[Path]:
    yield from sorted(path for path in root.rglob("*.png") if path.is_file())


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{size} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def optimize_png(
    path: Path, *, max_edge: int, min_bytes: int, dry_run: bool
) -> tuple[int, int, bool]:
    original_size = path.stat().st_size
    if original_size < min_bytes:
        return original_size, original_size, False

    with Image.open(path) as image:
        image.load()
        original_dimensions = image.size

        if max_edge > 0 and max(original_dimensions) > max_edge:
            image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            resized = image.size != original_dimensions
        else:
            resized = False

        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)

        try:
            image.save(tmp_path, format="PNG", optimize=True, compress_level=9)
            optimized_size = tmp_path.stat().st_size

            if optimized_size >= original_size:
                tmp_path.unlink()
                return original_size, original_size, resized

            if dry_run:
                tmp_path.unlink()
            else:
                os.replace(tmp_path, path)
            return original_size, optimized_size, resized
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Rendered site directory to scan")
    parser.add_argument(
        "--max-edge",
        type=int,
        default=3200,
        help="Downsample PNGs whose width or height exceeds this many pixels; use 0 for lossless-only optimization.",
    )
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=64 * 1024,
        help="Skip PNGs smaller than this many bytes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report savings without replacing image files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory")

    scanned = skipped = replaced = resized = 0
    total_before = total_after = 0

    for path in iter_pngs(root):
        scanned += 1
        before, after, did_resize = optimize_png(
            path,
            max_edge=args.max_edge,
            min_bytes=args.min_bytes,
            dry_run=args.dry_run,
        )
        total_before += before
        total_after += after
        if before == after:
            skipped += 1
            continue

        replaced += 1
        resized += int(did_resize)

    saved = total_before - total_after
    action = "Would optimize" if args.dry_run else "Optimized"
    print(
        f"{action} {replaced}/{scanned} PNGs "
        f"({resized} resized, {skipped} unchanged). "
        f"Saved {format_bytes(saved)}: {format_bytes(total_before)} -> {format_bytes(total_after)}."
    )


if __name__ == "__main__":
    main()
