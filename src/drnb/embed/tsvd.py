from dataclasses import dataclass

import sklearn.decomposition

from drnb.preprocess import center


@dataclass
class Tsvd:
    seed: int = None
    n_oversamples: int = 10
    n_iter: int = 5
    power_iteration_normalizer: str = "auto"

    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)

    def embed(self, x):
        return embed_tsvd(
            x,
            seed=self.seed,
            n_oversamples=self.n_oversamples,
            n_iter=self.n_iter,
            power_iteration_normalizer=self.power_iteration_normalizer,
        )


def embed_tsvd(
    x,
    seed=None,
    n_oversamples=10,
    n_iter=5,
    power_iteration_normalizer="auto",
):
    x = center(x)

    embedder = sklearn.decomposition.TruncatedSVD(
        random_state=seed,
        n_components=2,
        n_oversamples=n_oversamples,
        n_iter=n_iter,
        power_iteration_normalizer=power_iteration_normalizer,
    )
    embedded = embedder.fit_transform(x)

    return embedded
