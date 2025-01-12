import pynndescent.distances

from drnb.types import DistanceFunc


def distance_function(name: str) -> DistanceFunc:
    """Get a distance function by name. The function must be implemented in
    pynndescent.distances."""
    if hasattr(pynndescent.distances, name):
        return getattr(pynndescent.distances, name)
    raise ValueError(f"No distance function '{name}'.")
