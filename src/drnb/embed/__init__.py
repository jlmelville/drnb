import abc
from dataclasses import dataclass, field

from drnb.log import log


# helper method to create an embedder configuration
def embedder(name, params=None, **kwargs):
    return (name, kwargs | dict(params=params))


def check_embed_method(method, params=None):
    # in most cases you pass the method name and params to pass to the embedder
    # or a list of chained pre-computed embedder config
    if not isinstance(method, list):
        # or a pre-computed embedder config to allow for drnb keywords
        if not isinstance(method, tuple):
            method = embedder(method, params=params)
    elif params is not None:
        raise ValueError("params must be None when chained embedder provided")
    return method


def get_embedder_name(method):
    if isinstance(method, list):
        return "+".join(get_embedder_name(m) for m in method)
    if isinstance(method, tuple):
        if len(method) != 2:
            raise ValueError("Unexpected format for method")
        return method[0]
    return method


@dataclass
class Embedder(abc.ABC):
    params: dict = field(default_factory=dict)

    def embed(self, x, ctx=None):
        params = dict(self.params)
        return self.embed_impl(x, params, ctx)

    @abc.abstractmethod
    def embed_impl(self, x, params, ctx=None):
        pass


def get_coords(embedded):
    if isinstance(embedded, tuple):
        coords = embedded[0]
    elif isinstance(embedded, dict):
        coords = embedded["coords"]
    else:
        coords = embedded
    return coords


def run_embed(x, params, ctor, name):
    log.info("Running %s", name)
    embedder_ = ctor(
        **params,
    )
    embedded = embedder_.fit_transform(x)
    log.info("Embedding completed")

    return embedded
