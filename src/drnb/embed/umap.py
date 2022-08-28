from dataclasses import dataclass

import numpy as np
import umap


@dataclass
class Umap:
    n_neighbors: int = 15
    init: str = "spectral"
    densmap: bool = False
    seed: int = None

    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)

    def embed(self, x):
        return embed_umap(
            x,
            n_neighbors=self.n_neighbors,
            init=self.init,
            densmap=self.densmap,
            seed=self.seed,
        )


def embed_umap(
    x,
    n_neighbors=15,
    init="spectral",
    densmap=False,
    seed=None,
):
    umap_kwargs = {}
    if isinstance(x, np.ndarray) and x.shape[0] == x.shape[1]:
        umap_kwargs["metric"] = "precomputed"

    #         output_dens=densmap,
    embedder = umap.UMAP(
        random_state=seed,
        n_neighbors=n_neighbors,
        init=init,
        densmap=densmap,
        **umap_kwargs,
    )
    embedded = embedder.fit_transform(x)

    return embedded


# def umap_data(
#     name,
#     plot_kwargs=None,
#     export=False,
#     export_dir="umap",
#     seed=None,
#     repickle=False,
#     suffix=None,
#     init="spectral",
#     densmap=False,
#     n_neighbors=15,
#     x=None,
#     y=None,
# ):
#     if x is None:
#         x, y = read_dataxy(name, repickle=repickle)
#     if y is None:
#         y = range(x.shape[0])

#     embedded = umapper(
#         x,
#         y=y,
#         plot_kwargs=plot_kwargs,
#         init=init,
#         densmap=densmap,
#         seed=seed,
#         n_neighbors=n_neighbors,
#     )
#     if isinstance(embedded, tuple):
#         coords = embedded[0]
#     else:
#         coords = embedded

#     if export:
#         if suffix is None:
#             suffix = export_dir
#         if not suffix[0] in ("-", "_"):
#             suffix = f"-{suffix}"
#         write_csv(coords, name=f"{name}{suffix}", sub_dir=export_dir)
#         if densmap:
#             write_csv(embedded[1], name=f"{name}{suffix}-ro", sub_dir=export_dir)
#             write_csv(embedded[2], name=f"{name}{suffix}-re", sub_dir=export_dir)
#     return embedded


# DENSMAP


# def densmap_data(
#     name,
#     plot_kwargs=None,
#     export=False,
#     export_dir="densmap",
#     seed=None,
#     repickle=False,
#     suffix=None,
#     init="spectral",
#     n_neighbors=30,
#     x=None,
#     y=None,
# ):
#     return umap_data(
#         name=name,
#         plot_kwargs=plot_kwargs,
#         export=export,
#         export_dir=export_dir,
#         seed=seed,
#         repickle=repickle,
#         suffix=suffix,
#         init=init,
#         densmap=True,
#         n_neighbors=n_neighbors,
#         x=x,
#         y=y,
#     )
