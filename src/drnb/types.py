from typing import Callable

import numpy as np
import pandas as pd

# Used to define an action with an optional configuration
# e.g. for embedding: "tsne", ("tsne", {"n_components": 2})
# or data export: "csv", ("csv", {"create_sub_dir": False})
# some actions may be optional (e.g. scaling) in which case None should be used
ActionConfig = str | tuple[str, dict]
DataSet = tuple[np.ndarray, pd.DataFrame]
DistanceFunc = Callable[[np.float32, np.float32], np.float32]
EmbedResult = tuple | dict | np.ndarray
