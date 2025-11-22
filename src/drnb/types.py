from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

# Used to define an action with an optional configuration
# e.g. for embedding: "tsne", ("tsne", {"n_components": 2})
# or data export: "csv", ("csv", {"create_sub_dir": False})
# some actions may be optional (e.g. scaling) in which case None should be used
ActionConfig = str | tuple[str, dict]
DataSet = tuple[np.ndarray, pd.DataFrame]
DistanceFunc = Callable[[np.ndarray, np.ndarray], np.float32]
EmbedResult = tuple | dict | np.ndarray


@dataclass
class EmbedConfig:
    """Configuration for creating an embedder.

    This dataclass provides an explicit representation of embedder configurations.

    Attributes:
        name: The name of the embedder (e.g., "umap", "tsne").
        params: Parameters passed to the underlying embedding implementation's constructor.
        wrapper_kwds: drnb-specific wrapper options (e.g., use_precomputed_knn, initialization).
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    wrapper_kwds: dict[str, Any] = field(default_factory=dict)
