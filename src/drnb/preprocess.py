from typing import List, Literal

import numpy as np
import pandas as pd
from numpy.typing import DTypeLike
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from drnb.log import log
from drnb.types import ActionConfig
from drnb.util import get_method_and_args


def numpyfy(
    data: pd.DataFrame | np.ndarray,
    dtype: DTypeLike | None = None,
    layout: Literal["c", "f"] | None = None,
) -> np.ndarray:
    """Convert the data to a numpy array with the specified dtype and layout."""
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


def scale_data(
    data: np.ndarray | pd.DataFrame,
    scale_action: ActionConfig | None = None,
) -> np.ndarray | pd.DataFrame:
    """Scale the data according to the specified method. If scale_type is None or empty,
    no scaling is done and data is returned unchanged. Otherwise, the act of scaling
    may convert input data in DataFrame format into a numpy ndarray. For method-specific
    parameters, pass them in the `params` dictionary."""
    if scale_action is None or not scale_action:
        log.info("No scaling")
        return data

    scale_type, params = get_method_and_args(scale_action)

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


def center(data: np.ndarray) -> np.ndarray:
    """Center data by subtracting the mean of each column."""
    return data - np.mean(data, axis=0)


def zscale(data: np.ndarray) -> np.ndarray:
    """Scale data using the standard scaler."""
    return StandardScaler().fit_transform(data)


def range_scale(data: np.ndarray, minval: int = 0, maxval: int = 1) -> np.ndarray:
    """Scale data to the range [minval, maxval]."""
    return MinMaxScaler(feature_range=(minval, maxval)).fit_transform(data)


def robust_scale(data: np.ndarray) -> np.ndarray:
    """Scale data using the robust scaler."""
    return RobustScaler().fit_transform(data)


def filter_columns(
    data: np.ndarray | pd.DataFrame, cols: List[int] | List[str] | None
) -> np.ndarray | pd.DataFrame:
    """Filter the columns of the data to keep only the ones in `cols`.
    If `cols` is None, keep all columns. If `data` is a pandas DataFrame,
    it must be indexed by column name. If `data` is a numpy 2d array, it must be
    indexed by column integer index."""
    if cols is None:
        log.info("Keeping all columns")
        return data

    log.info("Keeping columns: %s", cols)
    if isinstance(data, pd.DataFrame):
        # pandas dataframe must be indexed by column name
        return data[cols]
    # better be a numpy 2d array indexed by column integer index
    return data[:, cols]


def normalize(data: np.ndarray, norm: Literal["l2", "l1", ""] = "") -> np.ndarray:
    """Normalize the data by dividing by the L1 or L2 norm of each row. If norm is
    empty, no normalization is done."""
    if norm == "l2":
        return normalize_l2(data)
    if norm == "l1":
        return normalize_l1(data)
    return data


def normalize_l1(data: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize the data by dividing by the L1 norm of each row"""
    return data / (np.sum(np.abs(data), axis=1)[:, np.newaxis] + eps)


def normalize_l2(data: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize the data by dividing by the L2 norm of each row"""
    return data / (np.linalg.norm(data, axis=1)[:, np.newaxis] + eps)


def pca(data: np.ndarray, n_components: int) -> np.ndarray:
    """Run PCA on the data and reduce to n_components."""
    log.info("Reducing initial dimensionality to %d", n_components)
    reducer = PCA(n_components=n_components).fit(data)
    varex = float(np.sum(reducer.explained_variance_ratio_) * 100.0)
    log.info(
        "PCA: %d components explain %.2f%% of variance",
        n_components,
        varex,
    )
    return reducer.transform(data)
