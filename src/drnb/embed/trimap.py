from dataclasses import dataclass

import trimap

import drnb.embed
from drnb.log import log


@dataclass
class Trimap(drnb.embed.Embedder):
    init: str = None
    return_every: int = None

    def embed_impl(self, x, params, ctx=None):
        return trimap_embed(x, params, self.init, self.return_every)


# pylint: disable=protected-access
def trimap_embed(x, params, init, return_every):
    # these settings just for the (rare case) when return_seq and return_every are set
    orig_return_every = trimap.trimap_._RETURN_EVERY
    if return_every is None:
        return_every = orig_return_every

    if params.get("return_seq", False):
        # https://github.com/eamid/trimap/issues/23
        # setting return_seq=True and init to "pca" or "random" will cause a crash
        if isinstance(init, str):
            init = None
            log.info("Setting init=None to avoid crash with return_seq=True")

    log.info("Running TriMap")
    # This is in a try block just to try really hard to put the internals of trimap
    # back to how we found them even if an exception gets thrown
    try:
        if return_every != orig_return_every:
            trimap.trimap_._RETURN_EVERY = return_every
        embedder = trimap.TRIMAP(n_dims=2, **params)
        result = embedder.fit_transform(x, init=init)

        if params.get("return_seq", False):
            embedded = dict(coords=result[:, :, -1], snapshots={})
            for i in range(result.shape[-1]):
                embedded["snapshots"][f"it_{return_every * i}"] = result[:, :, i]
        else:
            embedded = result
        log.info("Embedding completed")
        return embedded
    finally:
        # put _RETURN_EVERY back if necessary
        if trimap.trimap_._RETURN_EVERY != orig_return_every:
            trimap.trimap_._RETURN_EVERY = orig_return_every
