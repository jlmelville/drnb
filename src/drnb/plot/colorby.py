from __future__ import annotations

from typing import Any, Callable

from drnb.types import ActionConfig

ColorByBuilder = Callable[..., Any]
ColorByLike = ActionConfig | Callable[..., Any]


def _default_color_scale() -> dict:
    return {"palette": "Spectral"}


def _color_scale(color_scale: dict | None):
    from drnb.plot.scale import ColorScale

    return ColorScale.new(color_scale)


def color_by_ko(
    n_neighbors: int = 15,
    color_scale: dict | None = None,
    normalize: bool = True,
    log1p: bool = False,
):
    from drnb.plot.scale.ko import ColorByKo

    if color_scale is None:
        color_scale = _default_color_scale()

    return ColorByKo(
        n_neighbors,
        scale=_color_scale(color_scale),
        normalize=normalize,
        log1p=log1p,
    )


def color_by_so(
    n_neighbors: int = 15,
    color_scale: dict | None = None,
    normalize: bool = True,
    log1p: bool = False,
):
    from drnb.plot.scale.so import ColorBySo

    if color_scale is None:
        color_scale = _default_color_scale()

    return ColorBySo(
        n_neighbors,
        scale=_color_scale(color_scale),
        normalize=normalize,
        log1p=log1p,
    )


def color_by_lid(
    n_neighbors: int = 15,
    metric: str = "euclidean",
    color_scale: dict | None = None,
    remove_self: bool = True,
    eps: float = 1.0e-10,
    knn_params: dict | None = None,
):
    from drnb.plot.scale.lid import ColorByLid

    if color_scale is None:
        color_scale = _default_color_scale()

    return ColorByLid(
        n_neighbors=n_neighbors,
        metric=metric,
        scale=_color_scale(color_scale),
        remove_self=remove_self,
        eps=eps,
        knn_params=knn_params,
    )


def color_by_nbr_pres(
    n_neighbors: int = 15,
    normalize: bool = True,
    color_scale: dict | None = None,
    metric: str = "euclidean",
):
    from drnb.plot.scale.nbrpres import ColorByNbrPres

    if color_scale is None:
        color_scale = _default_color_scale()

    return ColorByNbrPres(
        n_neighbors,
        normalize=normalize,
        scale=_color_scale(color_scale),
        metric=metric,
    )


def color_by_rte(
    n_triplets_per_point: int = 5,
    normalize: bool = True,
    color_scale: dict | None = None,
    metric: str = "euclidean",
):
    from drnb.plot.scale.rte import ColorByRte

    if color_scale is None:
        color_scale = _default_color_scale()

    return ColorByRte(
        n_triplets_per_point=n_triplets_per_point,
        normalize=normalize,
        scale=_color_scale(color_scale),
        metric=metric,
    )


COLOR_BY_REGISTRY: dict[str, ColorByBuilder] = {
    "ko": color_by_ko,
    "so": color_by_so,
    "lid": color_by_lid,
    "nbrpres": color_by_nbr_pres,
    "rte": color_by_rte,
}


def create_color_by(color_by: ColorByLike):
    """Instantiate a color-by plot helper.

    Accepts:
    - callable instances (returned unchanged)
    - string names (looked up in COLOR_BY_REGISTRY)
    - (name, kwargs) tuples / ActionConfig entries
    """
    if callable(color_by) and not isinstance(color_by, str):
        return color_by

    name = color_by
    kwargs: dict[str, Any] = {}
    if isinstance(color_by, tuple) and len(color_by) == 2:
        name, kwargs = color_by
    if not isinstance(kwargs, dict):
        raise ValueError(f"Invalid color_by kwargs for {color_by}")

    name_str = str(name)
    builder = COLOR_BY_REGISTRY.get(name_str)
    if builder is None:
        raise ValueError(f"Unknown color_by '{name_str}'")
    return builder(**kwargs)
