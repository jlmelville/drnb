from dataclasses import dataclass
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap


@dataclass
class ColorScale:
    """Wrapper for a color scale function. When called, returns a ScalarMappable for
    use with a colorbar. If vmin or vmax are not set, they are set to the min and max
    of the data. If palette is not set, it is set to the palette of the color_by
    function."""

    vmin: float | None = None
    vmax: float | None = None
    palette: str | Colormap | None = None

    @classmethod
    def new(cls, kwds: dict | None) -> Self:
        """Create a new ColorScale object from a dictionary of keyword arguments."""
        if kwds is None:
            return cls()
        return cls(**kwds)

    def resolve(
        self,
        y: np.ndarray,
        vmin: float | None,
        vmax: float | None,
        palette: str | Colormap | None,
    ) -> tuple[float, float, str | Colormap | None]:
        """Resolve the effective vmin, vmax and palette for a set of values."""
        values = np.asarray(y)

        _vmin = self.vmin
        if _vmin is None:
            _vmin = vmin
        if _vmin is None:
            _vmin = values.min()

        _vmax = self.vmax
        if _vmax is None:
            _vmax = vmax
        if _vmax is None:
            _vmax = values.max()

        _palette = self.palette
        if _palette is None:
            _palette = palette

        return _vmin, _vmax, _palette

    def __call__(
        self,
        y: np.ndarray,
        vmin: float | None,
        vmax: float | None,
        palette: str | Colormap | None,
    ) -> plt.cm.ScalarMappable:
        _vmin, _vmax, _palette = self.resolve(y, vmin, vmax, palette)

        norm = plt.Normalize(_vmin, _vmax)
        sm = plt.cm.ScalarMappable(cmap=_palette, norm=norm)
        sm.set_array([])

        return sm
