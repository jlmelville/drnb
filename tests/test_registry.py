from pathlib import Path

import pytest

from drnb.plugins.registry import Registry


def test_registry_requires_plugins_toml(monkeypatch, tmp_path) -> None:
    root = tmp_path / "plugins-root"
    root.mkdir()
    monkeypatch.setenv("DRNB_PLUGINS_ROOT", str(root))

    with pytest.raises(FileNotFoundError, match="plugins.toml"):
        Registry()


def test_registry_validates_plugin_dir(monkeypatch, tmp_path) -> None:
    root = tmp_path / "plugins-root"
    root.mkdir()
    cfg = root / "plugins.toml"
    cfg.write_text('[plugins]\nfoo = { folder = "foo" }\n', encoding="utf-8")
    monkeypatch.setenv("DRNB_PLUGINS_ROOT", str(root))

    with pytest.raises(FileNotFoundError, match="foo"):
        Registry()


def test_registry_loads_plugins_toml(monkeypatch, tmp_path) -> None:
    root = tmp_path / "plugins-root"
    plugin_dir = root / "bar"
    plugin_dir.mkdir(parents=True)
    cfg = root / "plugins.toml"
    cfg.write_text(
        '[plugins]\nbar = { folder = "bar", runner = "uv run drnb-plugin-run.py" }\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("DRNB_PLUGINS_ROOT", str(root))

    registry = Registry()
    spec = registry.lookup("bar")
    assert spec is not None
    assert spec.plugin_dir == plugin_dir.resolve()
    assert spec.runner == ["uv", "run", "drnb-plugin-run.py"]


def test_registry_validates_env_var_path_exists(monkeypatch, tmp_path) -> None:
    non_existent = tmp_path / "does-not-exist"
    monkeypatch.setenv("DRNB_PLUGINS_ROOT", str(non_existent))

    with pytest.raises(FileNotFoundError, match="does not exist"):
        Registry()


def test_registry_validates_env_var_path_is_directory(monkeypatch, tmp_path) -> None:
    not_a_dir = tmp_path / "not-a-dir"
    not_a_dir.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("DRNB_PLUGINS_ROOT", str(not_a_dir))

    with pytest.raises(NotADirectoryError, match="is not a directory"):
        Registry()
