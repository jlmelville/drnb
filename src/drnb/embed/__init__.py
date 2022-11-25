import abc
from dataclasses import dataclass, field


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
