import abc
from dataclasses import dataclass, field


def get_embedder_name(method):
    if isinstance(method, tuple):
        if len(method) != 2:
            raise ValueError("Unexpected format for method")
        return method[0]
    return method


@dataclass
class Embedder(abc.ABC):
    embedder_kwds: dict = field(default_factory=dict)

    @abc.abstractmethod
    def embed(self, x):
        pass


def get_coords(embedded):
    if isinstance(embedded, tuple):
        coords = embedded[0]
    elif isinstance(embedded, dict):
        coords = embedded["coords"]
    else:
        coords = embedded
    return coords
