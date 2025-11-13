import numpy as np
import umato

import drnb.embed
import drnb.embed.base
from drnb.embed.context import EmbedContext
from drnb.log import log
from drnb.types import EmbedResult

DEFAULT_HUB_NUM = 300


class Umato(drnb.embed.base.Embedder):
    """UMATO embedding."""

    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> EmbedResult:
        # UMATO default hub_num is 300 and will fail if there are fewer than 300
        # samples in the dataset. In this case, we will (somewhat arbitrarily) reduce the
        # hub_num to data set size / 3 (https://github.com/hyungkwonko/umato/issues/21)
        if x.shape[0] < DEFAULT_HUB_NUM:
            small_hub_num = x.shape[0] // 3
            hub_num = params.get("hub_num", DEFAULT_HUB_NUM)
            if hub_num > x.shape[0]:
                log.warning("Forcing hub_num to %d", small_hub_num)
                params["hub_num"] = small_hub_num

        return drnb.embed.fit_transform_embed(x, params, umato.UMATO, "UMATO")
