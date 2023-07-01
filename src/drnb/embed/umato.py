import umato

import drnb.embed
from drnb.log import log

DEFAULT_HUB_NUM = 300


class Umato(drnb.embed.Embedder):
    def embed_impl(self, x, params, ctx=None):
        return umato_embed(x, params)


def umato_embed(x, params):
    log.info("Running UMATO")
    # UMATO default hub_num is 300 and will fail if there are fewer than 300
    # samples in the dataset. In this case, we will (somewhat arbitrarily) reduce the
    # hub_num to data set size / 3 (https://github.com/hyungkwonko/umato/issues/21)
    if x.shape[0] < DEFAULT_HUB_NUM:
        small_hub_num = x.shape[0] // 3
        hub_num = params.get("hub_num", DEFAULT_HUB_NUM)
        if hub_num > x.shape[0]:
            log.warning("Forcing hub_num to %d", small_hub_num)
            params["hub_num"] = small_hub_num
    embedder = umato.UMATO(**params)
    embedded = embedder.fit_transform(x)
    log.info("Embedding completed")

    return embedded
