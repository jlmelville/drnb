import pynndescent.distances


def distance_function(name):
    if hasattr(pynndescent.distances, name):
        return getattr(pynndescent.distances, name)
    raise ValueError(f"No distance function '{name}'.")
