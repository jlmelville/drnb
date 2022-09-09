import collections.abc


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
