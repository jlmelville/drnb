import collections.abc
import datetime
import json
from typing import Any

# pylint: disable=unused-import
# import json_fix
import numpy as np
import pandas as pd

from drnb.log import log
from drnb.types import ActionConfig


def get_method_and_args(
    method: ActionConfig, default: dict | None = None
) -> tuple[str, dict]:
    """Get the method and arguments from the given tuple or string. If the method is a
    tuple, the second element is assumed to be the keyword arguments. Returns the method
    name and the keyword arguments (which could be empty)."""
    if default is None:
        default = {}
    kwds = default
    if isinstance(method, tuple):
        if len(method) != 2:
            raise ValueError("Unexpected format for method")
        kwds = method[1]
        method = method[0]
    return method, kwds


def islisty(o: Any) -> bool:
    """Check if the object is a non-string iterable."""
    return not isinstance(o, str) and isinstance(o, collections.abc.Iterable)


# pylint: disable=too-few-public-methods
class FromDict:
    """Mixin class to create an object from a dictionary."""

    @classmethod
    def new(cls, **kwargs):
        """Create a new object from the given keyword arguments."""
        return cls(**kwargs)


DATETIME_FMT = "%Y%m%d%H%M%S"
READABLE_DATETIME_FMT = "%Y-%m-%d %H:%M:%S"


def dts_now() -> float:
    """Return the current UTC timestamp as a float.

    Returns:
        float: Current UTC timestamp in seconds since the Unix epoch.
    """
    return dt_now().timestamp()


def dt_now() -> datetime.datetime:
    """Return the current UTC datetime.

    Returns:
        datetime.datetime: Current datetime object with UTC timezone.
    """
    return datetime.datetime.now(datetime.timezone.utc)


def dts_to_str(dts: float | None = None, fmt: str = DATETIME_FMT) -> str:
    """Convert a UTC timestamp to a formatted string.

    Args:
        dts: UTC timestamp in seconds since the Unix epoch. If None, uses current time.
        fmt: String format to use (default: DATETIME_FMT).

    Returns:
        str: Formatted datetime string.
    """
    if dts is None:
        dts = dts_now()
    return dts_to_dt(dts).strftime(fmt)


def dts_to_dt(dts: float) -> datetime.datetime:
    """Convert a UTC timestamp to a datetime object.

    Args:
        dts: UTC timestamp in seconds since the Unix epoch.

    Returns:
        datetime.datetime: Datetime object with UTC timezone.
    """
    return datetime.datetime.fromtimestamp(dts, tz=datetime.timezone.utc)


def categorize(df: pd.DataFrame, colname: str):
    """Convert the column colname in the DataFrame df to a Pandas category."""
    df[colname] = df[colname].astype("category")


# convert the numpy array of integer codes or Pandas series to a Pandas category series with name
# col_name using the list of category_names
def codes_to_categories(
    y: pd.Series | np.ndarray, category_names: list[str], col_name: str
) -> pd.Series:
    """Convert the numpy array of integer codes or Pandas series to a Pandas category series with name
    col_name using the list of category_names."""
    return pd.Series(
        list(map(category_names.__getitem__, y.astype(int))),
        name=col_name,
        dtype="category",
    )


def evenly_spaced(s: list, n: int) -> list:
    """Return a list of n items evenly spaced from the sequence s"""
    if n > len(s):
        raise ValueError(f"Can't return {n} items, length is {len(s)}")
    idxs = np.round(np.linspace(0, len(s) - 1, n)).astype(int).tolist()
    return [s[i] for i in idxs]
