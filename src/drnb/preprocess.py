import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def numpyfy(x, dtype=None, layout=None):
    # pandas
    if hasattr(x, "to_numpy"):
        x = x.to_numpy(dtype=dtype)
    if dtype is not None and x.dtype != dtype:
        x = x.astype(dtype)
    if layout is not None:
        if layout == "c" and not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)
        elif layout == "f" and not x.flags["F_CONTIGUOUS"]:
            x = np.asfortranarray(x)
    return x


def center(X):
    return X - np.mean(X, axis=0)


def zscale(X):
    return StandardScaler().fit_transform(X)


def range_scale(X, minval=0, maxval=1):
    return MinMaxScaler(feature_range=(minval, maxval)).fit_transform(X)
