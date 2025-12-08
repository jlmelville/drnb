from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from drnb.embed.context import EmbedContext
from drnb.plot.scale import ColorScale
from drnb.types import EmbedResult


@dataclass
class FixedVectorColorBy:
    """Treat a precomputed numeric vector as a color-by input."""

    values: np.ndarray
    scale: ColorScale = field(default_factory=ColorScale)
    label: str | None = None

    def __call__(
        self,
        _: np.ndarray,
        __: pd.DataFrame | pd.Series | np.ndarray | range | None,
        ___: np.ndarray,
        ____: EmbedContext | None = None,
    ) -> np.ndarray:
        return self.values

    def __str__(self) -> str:
        if self.label is not None:
            return self.label
        return "color-by-vector"


def normalize_color_values(
    values: np.ndarray | Sequence[float], n_points: int
) -> np.ndarray:
    """Convert color values to a numpy array and validate the length."""
    arr = np.asarray(values)
    if arr.shape[0] != n_points:
        raise ValueError(
            f"Length mismatch: {arr.shape[0]} color values for {n_points} embedded points"
        )
    return arr


def get_ctx_from_embed_result(
    embed_result: EmbedResult | dict[str, object],
) -> EmbedContext | None:
    """Extract an EmbedContext from a pipeline result dict if present."""
    if isinstance(embed_result, dict):
        ctx = embed_result.get("context")
        if isinstance(ctx, EmbedContext):
            return ctx
    return None
