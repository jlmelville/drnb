from dataclasses import dataclass

import pacmap


@dataclass
class Pacmap:
    n_neighbors: int = None
    init: str = "pca"
    seed: int = None

    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)

    def embed(self, x):
        return embed_pacmap(
            x,
            n_neighbors=self.n_neighbors,
            init=self.init,
            seed=self.seed,
        )


def embed_pacmap(x, n_neighbors=None, init="pca", seed=None):

    pacmap_reducer = pacmap.PaCMAP(
        n_components=2, n_neighbors=n_neighbors, random_state=seed
    )
    pacmap_embed = pacmap_reducer.fit_transform(x, init=init)

    return pacmap_embed


# def pacmap_data(
#     name,
#     pacmap_kwargs=None,
#     plot_kwargs=None,
#     export=False,
#     export_dir="pacmap",
#     export_triplets=False,
#     seed=None,
#     n_neighbors=None,
#     n_iters=450,
#     suffix=None,
#     intermediates=None,
#     repickle=False,
# ):
#     x, y = read_dataxy(name, repickle=repickle)

#     #    Using int seed seems to give horrible results with PaCMAP
#     if pacmap_kwargs is None:
#         pacmap_kwargs = PACMAP_DEFAULTS

#     embedded = pacmapper(
#         x,
#         y=y,
#         pacmap_kwargs=pacmap_kwargs.update(
#             n_neighbors=n_neighbors, num_iters=n_iters, intermediate=intermediates
#         ),
#         plot_kwargs=plot_kwargs,
#     )
#     if export_triplets:
#         triplets = get_triplets(x, seed=seed)
#         write_csv(
#             np.vstack(triplets), name=f"{name}{suffix}-triplets", sub_dir=export_dir
#         )
#     else:
#         triplets = None
#     # if intermediates is not None and intermediates:
#     #     print(random_triplet_eval(x.to_numpy(), embedded[-1, :, :], triplets=triplets))
#     # else:
#     #     print(random_triplet_eval(x.to_numpy(), embedded, triplets=triplets))

#     if suffix is None:
#         suffix = export_dir
#     if suffix[0] not in ("-", "_"):
#         suffix = f"-{suffix}"

#     if export:
#         if intermediates is not None and intermediates:
#             for i, inter in enumerate(intermediates):
#                 write_csv(
#                     embedded[i, :, :],
#                     name=f"{name}{suffix}-it{inter}",
#                     sub_dir=export_dir,
#                 )
#         else:
#             write_csv(embedded, name=f"{name}{suffix}", sub_dir=export_dir)
#     return embedded
