def test_neighbors_and_plugin_imports() -> None:
    import drnb.neighbors  # noqa: F401
    import drnb.neighbors.compute  # noqa: F401
    import drnb.neighbors.nbrinfo  # noqa: F401
    import drnb.neighbors.store  # noqa: F401
    import drnb.nnplugins.external  # noqa: F401
