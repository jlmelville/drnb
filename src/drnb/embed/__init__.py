from typing import Any

import numpy as np

from drnb.log import log
from drnb.types import EmbedConfig, EmbedResult


# helper method to create an embedder configuration
# An embedder can be as simple as:
# embedder("tsne")
# or as complex as:
# embedder(
#     "tsne",
#     affinity="uniform",
#     n_neighbors=10,
#     anneal_exaggeration=True,
#     params=dict(n_iter=2000),
# )
def embedder(name: str, params: dict | None = None, **kwargs) -> EmbedConfig:
    """Create an embedder configuration.
    `name` is the name of the embedder, e.g. "tsne"
    `params` is an optional dictionary of parameters which will be passed to the
    underlying implementation, e.g. the constructor of a scikit-learn class.
    `kwargs` are additional parameters which will be passed to drnb's embedder
    class which wraps the underlying implementation and which attempts to provide an
    interface that is consistent across different implementations for common
    functionality such as pre-computed nearest neighbors and initialized coordinates.

    An embedder can be as simple as:
    `embedder("tsne")`
    or as complex as:
    `embedder("tsne", affinity="uniform", n_neighbors=10, anneal_exaggeration=True, params={ "n_iter": 2000 })`
    """
    return EmbedConfig(
        name=name,
        params=params or {},
        wrapper_kwds=kwargs,
    )


def _normalize_to_embed_config(
    method: str | tuple | EmbedConfig, params: dict | None = None
) -> EmbedConfig:
    """Normalize old format (str, tuple) into EmbedConfig.

    This helper converts various input formats into a single EmbedConfig representation.
    If method is already an EmbedConfig, merges params with override semantics.
    """
    if isinstance(method, EmbedConfig):
        # Already an EmbedConfig, merge params if provided
        if params is not None:
            method = EmbedConfig(
                name=method.name,
                params=method.params | params,  # params argument overrides
                wrapper_kwds=method.wrapper_kwds,
            )
        return method

    if isinstance(method, tuple):
        if len(method) != 2:
            raise ValueError("Unexpected format for method")
        name, config_dict = method
        # Extract params from config_dict if present
        config_params = config_dict.pop("params", None) or {}
        # Merge with params argument (params argument takes precedence)
        if params is not None:
            merged_params = config_params | params
        else:
            merged_params = config_params
        # Remaining items in config_dict are wrapper_kwds
        return EmbedConfig(
            name=name, params=merged_params, wrapper_kwds=config_dict.copy()
        )

    # String input
    return embedder(method, params=params or {})


def check_embed_method(
    method: str | list | tuple | EmbedConfig, params: dict | None = None
) -> tuple | list | EmbedConfig:
    """Ensure that the embedder method is in the correct format. Chained embedders
    can be provided as a list of pre-computed embedder configurations, but in this case
    the params must be None.

    If the `method` is a tuple (e.g. one created by `embedder`), then any embedder
    params contained within the tuple will be merged with the `params` argument. If
    there are conflicts between the two, values in the `params` argument will take
    precedence, i.e. the result of calling this:

    ```
    check_embed_method(
    embedder(
        "pacmap", local_scale=False, params=dict(n_neighbors=15, apply_pca=True)
    ),
    params=dict(apply_pca=False),
    ```

    will return an EmbedConfig with merged params.
    """
    if isinstance(method, list):
        if params is not None:
            raise ValueError("params must be None when chained embedder provided")
        # Normalize each element in the list
        normalized = [_normalize_to_embed_config(m) for m in method]
        # For backward compatibility, convert back to list of tuples
        return [
            (cfg.name, {**cfg.wrapper_kwds, "params": cfg.params}) for cfg in normalized
        ]

    # Normalize to EmbedConfig
    normalized = _normalize_to_embed_config(method, params)
    # For backward compatibility, convert back to tuple format
    return (normalized.name, {**normalized.wrapper_kwds, "params": normalized.params})


def get_embedder_name(method: list[str] | tuple | str | EmbedConfig) -> str:
    """Get the name of the embedder."""
    # chained embedder is a list of embedder names
    if isinstance(method, list):
        return "+".join(get_embedder_name(m) for m in method)
    if isinstance(method, EmbedConfig):
        return method.name
    if isinstance(method, tuple):
        # method is either just the string name or a tuple of (name, params)
        if len(method) != 2:
            raise ValueError("Unexpected format for method")
        return str(method[0])
    return str(method)


def get_coords(embedded: EmbedResult) -> np.ndarray:
    """Get the coordinates from the embedded data."""
    if isinstance(embedded, tuple):
        coords = embedded[0]
    elif isinstance(embedded, dict):
        coords = embedded["coords"]
    else:
        coords = embedded
    return coords


def set_coords(embedded: EmbedResult, coords: np.ndarray) -> EmbedResult:
    """Set the coordinates in the embedded data."""
    if isinstance(embedded, tuple):
        embedded = (coords, *embedded[1:])
    elif isinstance(embedded, dict):
        embedded["coords"] = coords
    else:
        embedded = coords
    return embedded


def fit_transform_embed(
    x: np.ndarray,
    params: dict,
    ctor: Any,
    name: str,
    fit_transform_params: dict | None = None,
    **kwargs,
) -> np.ndarray:
    """Create an embedder via `ctor` and its `params`, run the embedding and return the
    embedded coordinates. The embedder must have a `fit_transform` method that takes
    the data `x` and returns the embedded coordinates. Any kwargs are passed to the
    embedder's constructor along with the params. If `fit_transform_params` is provided,
    these are passed to the `fit_transform` method of the embedder. The `name` is used
    for logging.

    This is a convenience function which removes a small amount of boilerplate code
    from Embedders which follow the sklearn API and return the embedded coordinates
    only."""
    # add kwargs to params
    params |= kwargs
    log.info("Running %s", name)
    embedder_ = ctor(
        **params,
    )

    if fit_transform_params is None:
        fit_transform_params = {}
    fit_transform_params |= {"X": x}
    embedded = embedder_.fit_transform(**fit_transform_params)
    log.info("Embedding completed")

    return embedded
