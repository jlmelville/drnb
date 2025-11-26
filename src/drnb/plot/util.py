import matplotlib
import numpy as np
import pandas as pd


def rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    """Convert an RGB tuple of floats in (0, 1) to a hex code."""
    return matplotlib.colors.to_hex(rgb)


def hex_to_rgb(
    hexcode: str, scale: bool = False
) -> tuple[float, float, float] | tuple[int, int, int]:
    """Convert a hex code to an RGB tuple in (0.0, 1.0) if scale is True or (0, 255)
    otherwise."""
    result = tuple(int(hexcode[i : i + 2], 16) for i in (1, 3, 5))
    if scale:
        result = tuple(x / 255 for x in result)
    return result


# https://stackoverflow.com/a/67001213
def is_string_series(s: pd.Series) -> bool:
    """Check if a series is a string series."""
    if isinstance(s.dtype, pd.StringDtype):
        # The series was explicitly created as a string series (Pandas>=1.0.0)
        return True
    if s.dtype == "object":
        # Object series, check each value
        return all((v is None) or isinstance(v, str) for v in s)
    return False


def is_hex(col: list | pd.Series | range | None) -> bool:
    """Check if a column is a hex color column."""
    if isinstance(col, (list, tuple)):
        col = pd.Series(col)
    if not isinstance(col, pd.Series):
        return False
    if not is_string_series(col):
        return False
    return np.all(
        col.str.startswith("#")
        & (col.str.len() == 7)
        & (col.str.slice(1).str.match(r"[A-Fa-f0-9]"))
    )
