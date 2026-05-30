from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_create_plugin_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "create_plugin.py"
    spec = importlib.util.spec_from_file_location("create_plugin", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_sdk_selects_python_313_sdk_from_version() -> None:
    create_plugin = load_create_plugin_module()

    sdk_name, requires_python, py_version = create_plugin.resolve_sdk(
        "3.13", None, was_python_explicit=True
    )

    assert sdk_name == "drnb-plugin-sdk-313"
    assert requires_python == ">=3.13,<3.14"
    assert py_version == "3.13"


def test_resolve_sdk_defaults_to_python_313_sdk() -> None:
    create_plugin = load_create_plugin_module()

    sdk_name, requires_python, py_version = create_plugin.resolve_sdk(
        create_plugin.DEFAULT_PYTHON_VERSION, None, was_python_explicit=False
    )

    assert sdk_name == "drnb-plugin-sdk-313"
    assert requires_python == ">=3.13,<3.14"
    assert py_version == "3.13"


def test_resolve_sdk_313_override_sets_python_default() -> None:
    create_plugin = load_create_plugin_module()

    sdk_name, requires_python, py_version = create_plugin.resolve_sdk(
        "3.12", "313", was_python_explicit=False
    )

    assert sdk_name == "drnb-plugin-sdk-313"
    assert requires_python == ">=3.13,<3.14"
    assert py_version == "3.13"


def test_resolve_sdk_312_override_sets_python_default() -> None:
    create_plugin = load_create_plugin_module()

    sdk_name, requires_python, py_version = create_plugin.resolve_sdk(
        create_plugin.DEFAULT_PYTHON_VERSION, "312", was_python_explicit=False
    )

    assert sdk_name == "drnb-plugin-sdk-312"
    assert requires_python == ">=3.12,<3.13"
    assert py_version == "3.12"


def test_resolve_sdk_310_uses_legacy_exact_requirement_with_minor_hint() -> None:
    create_plugin = load_create_plugin_module()

    sdk_name, requires_python, py_version = create_plugin.resolve_sdk(
        "3.10", None, was_python_explicit=True
    )

    assert sdk_name == "drnb-plugin-sdk-310"
    assert requires_python == "==3.10.14"
    assert py_version == "3.10"


def test_format_pyproject_uses_python_313_sdk_source_path() -> None:
    create_plugin = load_create_plugin_module()

    pyproject = create_plugin.format_pyproject(
        folder="demo",
        description="demo plugin runner for drnb",
        requires_python=">=3.13,<3.14",
        sdk_name="drnb-plugin-sdk-313",
        deps=[],
    )

    assert 'requires-python = ">=3.13,<3.14"' in pyproject
    assert '"drnb-plugin-sdk-313>=0.1.0",' in pyproject
    assert (
        'drnb-plugin-sdk-313 = { path = "../../plugin-sdks/drnb-plugin-sdk-313" }'
        in pyproject
    )
