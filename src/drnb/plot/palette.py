from typing import List

import glasbey
import numpy as np
import pandas as pd
import seaborn as sns

from drnb.plot.util import rgb_to_hex
from drnb.util import evenly_spaced


# use glasbey to extend a categorical palette if possible (or necessary)
def palettize(
    color_col: np.ndarray | pd.Series | range, palette: dict | str | None = None
) -> dict | List[str] | np.ndarray:
    """Create a categorical palette for a color column, using glasbey to extend it if
    possible (or necessary)."""

    # return early if the palette maps from category level to color
    if isinstance(palette, dict):
        return palette
    n_categories = None
    if pd.api.types.is_categorical_dtype(color_col):
        if hasattr(color_col, "categories"):
            n_categories = color_col.categories.nunique()
        else:
            n_categories = color_col.nunique()
    else:
        # if this isn't a categorical color column do nothing
        return palette

    if palette is None:
        # return a suitably sized categorical palette
        # returns a List[str] or ndarray
        return glasbey.create_palette(n_categories)

    # a named or pre-defined palette, so check we have enough colors
    if isinstance(palette, str):
        # list of rgb tuples [(r, g, b), ...] where r, g, b are float in [0, 1]
        palette = sns.color_palette(palette)

    # we must have a list of colors at this point
    # pylint: disable=protected-access
    if not (isinstance(palette, sns.palettes._ColorPalette) or isinstance(palette, (list, tuple))):
        raise ValueError(f"Unknown palette {palette}")

    n_colors = len(palette)
    if n_categories is not None:
        if n_colors < n_categories:
            # palette is now a list of hex codes
            palette = glasbey.extend_palette(palette, n_categories)
        elif n_colors > n_categories:
            palette = evenly_spaced(palette, n_categories)
        # if n_colors == n_categories then use the palette as-is

        # glasbey returns hex codes, but seaborn does not, convert to hex codes here
        if not isinstance(palette[0], str):
            palette = [rgb_to_hex(color) for color in palette]
    return palette
