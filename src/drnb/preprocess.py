import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from drnb.log import log
from drnb.util import get_method_and_args


def numpyfy(data, dtype=None, layout=None):
    # pandas
    if hasattr(data, "to_numpy"):
        data = data.to_numpy(dtype=dtype)
    if dtype is not None and data.dtype != dtype:
        data = data.astype(dtype)
    if layout is not None:
        if layout == "c" and not data.flags["C_CONTIGUOUS"]:
            data = np.ascontiguousarray(data)
        elif layout == "f" and not data.flags["F_CONTIGUOUS"]:
            data = np.asfortranarray(data)
    return data


def scale_data(data, scale_type=None, params=None):
    if scale_type is None or not scale_type:
        log.info("No scaling")
        return data

    if scale_type == "center":
        log.info("Centering")
        return center(data)
    if scale_type in ("z", "zscale", "standard"):
        log.info("Z-Scaling")
        return zscale(data)
    if scale_type in ("range", "minmax"):
        log.info("range scaling")
        return range_scale(data, **params)
    raise ValueError(f"Unknown scale type {scale_type}")


def create_scale_kwargs(scale):
    scale, kwds = get_method_and_args(scale)
    if kwds is None:
        kwds = {}
    kwds["scale_type"] = scale
    return kwds


def center(data):
    return data - np.mean(data, axis=0)


def zscale(data):
    return StandardScaler().fit_transform(data)


def range_scale(data, minval=0, maxval=1):
    return MinMaxScaler(feature_range=(minval, maxval)).fit_transform(data)


def filter_columns(data, cols):
    if cols is None and cols:
        return data

    log.info("Keeping columns: %s", cols)
    if isinstance(data, pd.DataFrame):
        # pandas dataframe must be indexed by column name
        return data[cols]
    # better be a numpy 2d array indexed by column integer index
    return data[:, cols]
