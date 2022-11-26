from dataclasses import dataclass

import numba
from umap.layouts import clip, rdist
from umap.utils import tau_rand_int

import drnb.embed.umap
from drnb.embed import run_embed
from drnb.embed.umap.custom import CustomGradientUMAP


# note that gamma has been renamed to z-bar in this function
# I reuse gamma internally to represent z-bar * m * xi from the paper
# pylint: disable=unused-argument
def negtsne_gradient_func(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    z_bar,
    dim,
    move_other,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
):
    xi = 2 / (n_vertices * (n_vertices - 1))
    # pylint: disable=not-an-iterable
    for i in numba.prange(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]

            current = head_embedding[j]
            other = tail_embedding[k]

            dist_squared = rdist(current, other)

            n_neg_samples = int(
                (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )

            gamma = z_bar * n_neg_samples * xi

            if dist_squared > 0.0:
                w = 1 / (1 + dist_squared)
                grad_coeff = -2.0 * gamma * w
                grad_coeff /= w + gamma
            else:
                grad_coeff = 0.0

            for d in range(dim):
                grad_d = clip(grad_coeff * (current[d] - other[d]))

                current[d] += grad_d * alpha
                if move_other:
                    other[d] += -grad_d * alpha

            epoch_of_next_sample[i] += epochs_per_sample[i]

            for _ in range(n_neg_samples):
                k = tau_rand_int(rng_state) % n_vertices
                if j == k:
                    continue

                other = tail_embedding[k]

                dist_squared = rdist(current, other)

                if dist_squared > 0.0:
                    w = 1 / (1 + dist_squared)
                    grad_coeff = 2.0 * w * w
                    grad_coeff /= w + gamma
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


class NegTSNE(CustomGradientUMAP):
    def __init__(self, **kwargs):
        if "anneal_lr" not in kwargs:
            kwargs["anneal_lr"] = False
        super().__init__(custom_epoch_func=negtsne_gradient_func, **kwargs)


@dataclass
class NegTsne(drnb.embed.umap.Umap):
    use_precomputed_knn: bool = True
    drnb_init: str = None

    def embed_impl(self, x, params, ctx=None):
        params = self.update_params(x, params, ctx)
        return run_embed(x, params, NegTSNE, "Neg-t-SNE")
