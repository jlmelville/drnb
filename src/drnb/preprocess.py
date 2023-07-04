from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

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


def scale_data(data, scale_type=None, params: Optional[dict] = None):
    if scale_type is None or not scale_type:
        log.info("No scaling")
        return data

    if params is None:
        params = {}

    if scale_type == "center":
        log.info("Centering")
        return center(data)
    if scale_type in ("z", "zscale", "standard"):
        log.info("Z-Scaling")
        return zscale(data)
    if scale_type in ("range", "minmax"):
        log.info("Range scaling")
        return range_scale(data, **params)
    if scale_type == "robust":
        log.info("Robust scaling")
        return robust_scale(data)
    raise ValueError(f"Unknown scale type {scale_type}")


def create_scale_kwargs(scale):
    scale, kwds = get_method_and_args(scale, {})
    kwds["scale_type"] = scale
    return kwds


def center(data):
    return data - np.mean(data, axis=0)


def zscale(data):
    return StandardScaler().fit_transform(data)


def range_scale(data, minval=0, maxval=1):
    return MinMaxScaler(feature_range=(minval, maxval)).fit_transform(data)


def robust_scale(data):
    return RobustScaler().fit_transform(data)


def filter_columns(data, cols):
    if cols is None:
        log.info("Keeping all columns")
        return data

    log.info("Keeping columns: %s", cols)
    if isinstance(data, pd.DataFrame):
        # pandas dataframe must be indexed by column name
        return data[cols]
    # better be a numpy 2d array indexed by column integer index
    return data[:, cols]


def normalize(data, norm=""):
    if norm == "l2":
        return normalize_l2(data)
    if norm == "l1":
        return normalize_l1(data)
    return data


def normalize_l1(data, eps=1e-8):
    return data / (np.sum(np.abs(data), axis=1)[:, np.newaxis] + eps)


def normalize_l2(data, eps=1e-8):
    return data / (np.linalg.norm(data, axis=1)[:, np.newaxis] + eps)


def pca(data, n_components):
    log.info("Reducing initial dimensionality to %d", n_components)
    reducer = PCA(n_components=n_components).fit(data)
    varex = float(np.sum(reducer.explained_variance_ratio_) * 100.0)
    log.info(
        "PCA: %d components explain %.2f%% of variance",
        n_components,
        varex,
    )
    return reducer.transform(data)
