from dataclasses import dataclass

import trimap

import drnb.embed
from drnb.log import log


@dataclass
class Trimap(drnb.embed.Embedder):
    def embed(self, x):
        return trimap_embed(x, self.embedder_kwds)


# pylint: disable=protected-access
def trimap_embed(x, embedder_kwds):
    if "init" in embedder_kwds:
        init = embedder_kwds["init"]
        del embedder_kwds["init"]
    else:
        init = None

    old_return_every = None
    new_return_every = None
    if embedder_kwds.get("return_seq", False):
        # https://github.com/eamid/trimap/issues/23
        # setting return_seq=True and init to "pca" or "random" will cause a crash
        if isinstance(init, str):
            init = None
            log.info("Setting init=None to avoid crash with return_seq=True")

        # Use an internal setting to change the number of snapshots we get
        if "_return_every" in embedder_kwds:
            old_return_every = trimap.trimap_._RETURN_EVERY
            new_return_every = embedder_kwds["_return_every"]
            del embedder_kwds["_return_every"]

    # This is in a try block just to try really hard to put the internals of trimap
    # back to how we found them even if an exception gets thrown
    try:
        if new_return_every is not None:
            trimap.trimap_._RETURN_EVERY = new_return_every
        embedder = trimap.TRIMAP(n_dims=2, **embedder_kwds)
        result = embedder.fit_transform(x, init=init)

        if embedder_kwds.get("return_seq", False):
            if new_return_every is not None:
                return_every = new_return_every
            else:
                return_every = trimap.trimap_._RETURN_EVERY
            embedded = dict(coords=result[:, :, -1])
            for i in range(result.shape[-1]):
                embedded[f"it_{return_every * i}"] = result[:, :, i]
        else:
            embedded = result

        return embedded
    finally:
        # put _RETURN_EVERY back if necessary
        if old_return_every is not None:
            trimap.trimap_._RETURN_EVERY = old_return_every
