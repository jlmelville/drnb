from dataclasses import dataclass

import numba
from umap.layouts import clip, rdist
from umap.utils import tau_rand_int

import drnb.embed.umap
from drnb.embed.umap.custom import CustomGradientUMAP
from drnb.log import log


# pylint: disable=unused-argument
def negtumap_gradient_func(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma,
    dim,
    move_other,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
):
    # pylint: disable=not-an-iterable
    for i in numba.prange(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]

            current = head_embedding[j]
            other = tail_embedding[k]

            dist_squared = rdist(current, other)

            if dist_squared > 0.0:
                w = 1 / (1 + dist_squared)
                grad_coeff = -2.0 * w
                grad_coeff /= w + 1.0
            else:
                grad_coeff = 0.0

            for d in range(dim):
                grad_d = clip(grad_coeff * (current[d] - other[d]))

                current[d] += grad_d * alpha
                if move_other:
                    other[d] += -grad_d * alpha

            epoch_of_next_sample[i] += epochs_per_sample[i]

            n_neg_samples = int(
                (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )

            for _ in range(n_neg_samples):
                k = tau_rand_int(rng_state) % n_vertices
                if j == k:
                    continue
                other = tail_embedding[k]

                dist_squared = rdist(current, other)

                if dist_squared > 0.0:
                    w = 1 / (1 + dist_squared)
                    grad_coeff = 2.0 * w * w
                    grad_coeff /= w + 1.0
                else:
                    grad_coeff = 0.0

                for d in range(dim):
                    if grad_coeff > 0.0:
                        grad_d = clip(grad_coeff * (current[d] - other[d]))
                    else:
                        grad_d = 4.0
                    current[d] += grad_d * alpha

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )


class NegTUMAP(CustomGradientUMAP):
    def __init__(self, **kwargs):
        super().__init__(custom_gradient_func=negtumap_gradient_func, **kwargs)


@dataclass
class NegTumap(drnb.embed.umap.Umap):
    use_precomputed_knn: bool = True
    drnb_init: str = None

    def embed_impl(self, x, params, ctx=None):
        params = self.update_params(x, params, ctx)
        return embed_negtumap(x, params)


def embed_negtumap(
    x,
    params,
):
    log.info("Running Neg-t-UMAP")
    embedder = NegTUMAP(
        **params,
    )
    embedded = embedder.fit_transform(x)
    log.info("Embedding completed")

    return embedded
