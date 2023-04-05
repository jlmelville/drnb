import umato

import drnb.embed
from drnb.log import log


class Umato(drnb.embed.Embedder):
    def embed_impl(self, x, params, ctx=None):
        return umato_embed(x, params)


def umato_embed(x, params):
    log.info("Running UMATO")
    embedder = umato.UMATO(**params)
    embedded = embedder.fit_transform(x)
    log.info("Embedding completed")

    return embedded
