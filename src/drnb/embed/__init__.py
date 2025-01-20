from typing import Any, List

import numpy as np

from drnb.log import log
from drnb.types import EmbedResult


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
def embedder(name: str, params: dict | None = None, **kwargs) -> tuple[str, dict]:
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
    return (name, kwargs | {"params": params})


def check_embed_method(
    method: str | list | tuple, params: dict | None = None
) -> str | list | tuple:
    """Ensure that the embedder method is in the correct format. Chained embedders
    can be provided as a list of pre-computed embedder configurations, but in this case
    the params must be None."""
    # in most cases you pass the method name and params to pass to the embedder
    # or a list of chained pre-computed embedder config
    if not isinstance(method, list):
        # or a pre-computed embedder config to allow for drnb keywords
        if isinstance(method, tuple):
            if len(method) != 2:
                raise ValueError("Unexpected format for method")
            method = embedder(method[0], params=params, **method[1])
        if not isinstance(method, tuple):
            method = embedder(method, params=params)
    elif params is not None:
        raise ValueError("params must be None when chained embedder provided")
    return method


def get_embedder_name(method: List[str] | tuple | str) -> str:
    """Get the name of the embedder."""
    # chained embedder is a list of embedder names
    if isinstance(method, list):
        return "+".join(get_embedder_name(m) for m in method)
    if isinstance(method, tuple):
        # method is either just the string name or a tuple of (name, params)
        if len(method) != 2:
            raise ValueError("Unexpected format for method")
        return method[0]
    return method


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
