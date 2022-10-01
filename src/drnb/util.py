import collections.abc
import datetime
import json
from dataclasses import asdict

# pylint: disable=unused-import
import json_fix
import pandas as pd


def get_method_and_args(method, default=None):
    kwds = default
    if isinstance(method, tuple):
        if len(method) != 2:
            raise ValueError("Unexpected format for method")
        kwds = method[1]
        method = method[0]
    return method, kwds


def islisty(o):
    return not isinstance(o, str) and isinstance(o, collections.abc.Iterable)


# normalize possible config list
# bool or string or (file_type, {options}) should be put in a list
def get_multi_config(config):
    if not islisty(config) or isinstance(config, tuple):
        return [config]
    return config


class Jsonizable:
    @property
    def __dict__(self):
        return asdict(self)

    def to_json(self, indent=None):
        return json.dumps(self.__dict__, indent=indent, ensure_ascii=False)

    def __json__(self):
        return self.__dict__


class FromDict:
    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)


DATETIME_FMT = "%Y%m%d%H%M%S"
READABLE_DATETIME_FMT = "%Y-%m-%d %H:%M:%S"


def dts_now():
    return dt_now().timestamp()


def dt_now():
    return datetime.datetime.now(datetime.timezone.utc)


def dts_to_str(dts=None, fmt=DATETIME_FMT):
    if dts is None:
        dts = dts_now()
    return dts_to_dt(dts).strftime(fmt)


def dts_to_dt(dts):
    return datetime.datetime.fromtimestamp(dts, tz=datetime.timezone.utc)


def categorize(df, colname):
    df[colname] = df[colname].astype("category")


# convert the numpy array of integer codes to a pandas category series with name
# col_name using the list of category_names
def codes_to_categories(y, category_names, col_name):
    return pd.Series(
        list(map(category_names.__getitem__, y.astype(int))),
        name=col_name,
        dtype="category",
    )
