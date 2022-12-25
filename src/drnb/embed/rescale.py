from dataclasses import dataclass

import numpy as np

import drnb.embed
from drnb.log import log
from drnb.yinit import scale_coords


# This exists purely to rescale the output of one embedder if using it as initialization
# into another in an embedding pipeline
@dataclass
class Rescale(drnb.embed.Embedder):
    precomputed_init: np.ndarray = None
    max_coord: float = 10.0

    # pylint:disable=unused-argument
    def embed_impl(self, x, params, ctx=None):
        max_coord = params.get("max_coord", self.max_coord)
        if self.precomputed_init is not None:
            coords = self.precomputed_init
        else:
            raise ValueError("No coordinates provided")

        return embed_rescale(coords, max_coord)


def embed_rescale(coords, max_coord):
    log.info("Rescaling")
    return scale_coords(coords, max_coord=max_coord)
