#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGINS_DIR = REPO_ROOT / "plugins"
SDK_ROOT = REPO_ROOT / "plugin-sdks"


def read_repo_python_version() -> str:
    path = REPO_ROOT / ".python-version"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return "3.12.0"


def normalize_python_version(raw: str) -> str:
    parts = raw.strip().split(".")
    if len(parts) >= 3:
        return ".".join(parts[:3])
    if len(parts) == 2:
        major, minor = parts
        if major == "3" and minor == "10":
            return "3.10.14"
        return f"{major}.{minor}.0"
    if len(parts) == 1 and parts[0]:
        return f"{parts[0]}.12.0" if parts[0] == "3" else f"{parts[0]}.0.0"
    return "3.12.0"


def resolve_sdk(
    python_version: str, sdk_override: str | None, *, was_python_explicit: bool
) -> tuple[str, str, str]:
    """
    Return (sdk_name, requires_python_spec, python_version).

    sdk_name: drnb-plugin-sdk-312 or drnb-plugin-sdk-310
    requires_python_spec: string for pyproject requires-python
    python_version: possibly adjusted version (e.g., defaulting to 3.10.14 when SDK 310 is forced)
    """
    sdk_name = "drnb-plugin-sdk-312"
    requires_python = ">=3.12"

    if sdk_override:
        if "310" in sdk_override:
            sdk_name = "drnb-plugin-sdk-310"
            if not was_python_explicit:
                python_version = "3.10.14"
            requires_python = f"=={python_version}"
        else:
            sdk_name = "drnb-plugin-sdk-312"
            requires_python = ">=3.12"
        return sdk_name, requires_python, python_version

    parts = python_version.split(".")
    if len(parts) >= 2 and parts[0] == "3" and parts[1] == "10":
        sdk_name = "drnb-plugin-sdk-310"
        requires_python = f"=={python_version}"
    return sdk_name, requires_python, python_version


def format_pyproject(
    *,
    folder: str,
    description: str,
    requires_python: str,
    sdk_name: str,
    deps: Iterable[str],
) -> str:
    deps_all = ["numpy", *deps, f"{sdk_name}>=0.1.0"]
    lines = [
        "[project]",
        f'name = "drnb-plugin-{folder}"',
        'version = "0.0.1"',
        f'description = "{description}"',
        f'requires-python = "{requires_python}"',
        "dependencies = [",
        *[f'    "{dep}",' for dep in deps_all],
        "]",
        "",
        "[build-system]",
        'requires = ["setuptools"]',
        'build-backend = "setuptools.build_meta"',
        "",
        "[tool.uv.sources]",
        f'{sdk_name} = {{ path = "../../plugin-sdks/{sdk_name}" }}',
        "",
    ]
    return "\n".join(lines)


def sanitize_function_name(method: str) -> str:
    clean = []
    for ch in method:
        if ch.isalnum():
            clean.append(ch.lower())
        else:
            clean.append("_")
    name = "".join(clean).strip("_")
    if not name:
        name = "method"
    if name[0].isdigit():
        name = f"m_{name}"
    return name


def format_runner_script(methods: list[str]) -> str:
    lines: list[str] = [
        "#!/usr/bin/env python",
        "from __future__ import annotations",
        "",
        "from typing import Any",
        "",
        "import numpy as np",
        "from drnb_plugin_sdk import protocol as sdk_protocol",
        "from drnb_plugin_sdk.helpers.logging import log, summarize_params",
        "from drnb_plugin_sdk.helpers.paths import (",
        "    resolve_init_path,",
        "    resolve_neighbors,",
        "    resolve_x_path,",
        ")",
        "from drnb_plugin_sdk.helpers.results import save_result_npz",
        "from drnb_plugin_sdk.helpers.runner import run_plugin",
        "",
    ]

    for idx, method in enumerate(methods):
        fn_name = f"run_{sanitize_function_name(method)}"
        lines.extend(
            [
                f"def {fn_name}(req: sdk_protocol.PluginRequest) -> dict[str, Any]:",
                "    x = np.load(resolve_x_path(req), allow_pickle=False)",
                "    params = dict(req.params or {})",
                "",
                "    init_path = resolve_init_path(req)",
                "    init = np.load(init_path, allow_pickle=False) if init_path else None",
                "    neighbors = resolve_neighbors(req)",
                "",
                f'    log(f"Running {method} with params={{summarize_params(params)}}")',
                f'    raise NotImplementedError("Implement {method} embedding here")',
                "    # Example:",
                '    # coords = np.asarray(x[:, :2], dtype=np.float32, order="C")',
                "    # return save_result_npz(req.output.result_path, coords)",
            ]
        )
        if idx != len(methods) - 1:
            lines.append("")

    lines.extend(
        [
            "",
            "",
            'if __name__ == "__main__":',
            "    run_plugin(",
            "        {",
        ]
    )
    for method in methods:
        fn_name = f"run_{sanitize_function_name(method)}"
        lines.append(f'            "{method}": {fn_name},')
    lines.extend(
        [
            "        }",
            "    )",
            "",
        ]
    )

    return "\n".join(lines)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_file(path: Path, content: str, *, force: bool, dry_run: bool) -> None:
    if path.exists() and not force:
        print(f"[skip] {path} already exists (use --force to overwrite)")
        return
    if dry_run:
        print(f"[dry-run] Would write {path}")
        return
    path.write_text(content, encoding="utf-8")
    print(f"[write] {path}")


def update_registry(
    *,
    methods: list[str],
    folder: str,
    runner: str | None,
    dry_run: bool,
) -> None:
    registry_path = PLUGINS_DIR / "plugins.toml"
    existing: set[str] = set()

    if registry_path.exists() and tomllib is not None:
        data = tomllib.loads(registry_path.read_text(encoding="utf-8"))
        plugins_table = data.get("plugins") or {}
        existing = {key.lower() for key in plugins_table.keys()}
    else:
        print(f"[warn] Registry file missing at {registry_path}; will create new one")

    new_entries = []
    added_methods: list[str] = []
    for method in methods:
        if method.lower() in existing:
            print(f"[skip] plugins.toml already has entry for '{method}'")
            continue
        entry = [f'[plugins."{method}"]', f'folder = "{folder}"']
        if runner:
            entry.append(f'runner = "{runner}"')
        new_entries.append("\n".join(entry))
        added_methods.append(method)

    if not new_entries:
        return

    block = "\n\n" + "\n\n".join(new_entries) + "\n"
    if dry_run:
        print(f"[dry-run] Would append to {registry_path}:{block}")
        return

    mode = "a" if registry_path.exists() else "w"
    with registry_path.open(mode, encoding="utf-8") as f:
        f.write(block)
    print(f"[update] plugins.toml entries added: {', '.join(added_methods)}")


def check_sdk_exists(sdk_name: str) -> None:
    sdk_dir = SDK_ROOT / sdk_name
    if not sdk_dir.exists():
        print(
            f"[warn] SDK path {sdk_dir} does not exist. "
            "Ensure the SDK workspace is present before syncing.",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create boilerplate files for a new drnb plugin runner."
    )
    parser.add_argument("folder", help="Plugin folder name under plugins/")
    parser.add_argument(
        "--method",
        action="append",
        dest="methods",
        help="Method name to register (repeatable). Defaults to the folder name.",
    )
    parser.add_argument(
        "--description",
        help='Project description for pyproject.toml (default: "<folder> plugin runner for drnb")',
    )
    parser.add_argument(
        "--python-version",
        dest="python_version",
        help="Python version to target (e.g., 3.12.8 or 3.10.14). Defaults to repo .python-version.",
    )
    parser.add_argument(
        "--sdk",
        choices=["310", "312", "drnb-plugin-sdk-310", "drnb-plugin-sdk-312"],
        help="Force SDK variant regardless of Python version.",
    )
    parser.add_argument(
        "--dep",
        action="append",
        dest="deps",
        default=[],
        help="Additional dependency to include (repeatable).",
    )
    parser.add_argument(
        "--runner",
        help="Custom runner command to record in plugins.toml (e.g., uv run --quiet ...).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print actions without writing files."
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing files."
    )

    args = parser.parse_args(argv)

    folder = args.folder.strip().strip("/ ")
    if not folder:
        parser.error("Folder name must be non-empty")

    methods = args.methods or [folder]
    description = args.description or f"{folder} plugin runner for drnb"

    repo_py = read_repo_python_version()
    was_python_explicit = args.python_version is not None
    py_version = normalize_python_version(args.python_version or repo_py)

    sdk_name, requires_python, py_version = resolve_sdk(
        py_version, args.sdk, was_python_explicit=was_python_explicit
    )
    check_sdk_exists(sdk_name)

    deps = args.deps or []

    plugin_dir = PLUGINS_DIR / folder
    if args.dry_run:
        print(f"[dry-run] Would create {plugin_dir}")
    else:
        ensure_directory(plugin_dir)

    pyproject_path = plugin_dir / "pyproject.toml"
    pyproject_text = format_pyproject(
        folder=folder,
        description=description,
        requires_python=requires_python,
        sdk_name=sdk_name,
        deps=deps,
    )
    write_file(pyproject_path, pyproject_text, force=args.force, dry_run=args.dry_run)

    python_version_path = plugin_dir / ".python-version"
    write_file(
        python_version_path, f"{py_version}\n", force=args.force, dry_run=args.dry_run
    )

    runner_path = plugin_dir / "drnb-plugin-run.py"
    runner_text = format_runner_script(methods)
    write_file(runner_path, runner_text, force=args.force, dry_run=args.dry_run)

    update_registry(
        methods=methods,
        folder=folder,
        runner=args.runner,
        dry_run=args.dry_run,
    )

    print("[done] Plugin scaffolding complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
