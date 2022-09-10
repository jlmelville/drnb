import collections.abc
import datetime
import json
from dataclasses import asdict


def get_method_and_args(method):
    kwds = None
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


def dtstamp():
    return datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
