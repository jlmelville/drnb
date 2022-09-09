import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ("numpyfy", dict(dtype="float32", layout="c"))
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


# "center"
def center(data):
    return data - np.mean(data, axis=0)


# "zscale"
def zscale(data):
    return StandardScaler().fit_transform(data)


# ("range_scale", dict(minval=0, maxval=10.0))
def range_scale(data, minval=0, maxval=1):
    return MinMaxScaler(feature_range=(minval, maxval)).fit_transform(data)
