from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from drnb.embed.context import EmbedContext
from drnb.util import Jsonizable

from .base import EmbeddingEval, EvalResult


@dataclass
class GlobalScore(EmbeddingEval, Jsonizable):
    """Compute the global structure preservation score of an embedding.

    The global structure preservation score compares the global structure
    preservation of an embedding against a PCA baseline. The score is
    computed as exp( -(gs_emb - gs_pca) / gs_pca ), where gs_emb is the
    global structure preservation error of the embedding, and gs_pca is
    the global structure preservation error of the PCA projection of the
    original data.

    A score above 1 indicates that the embedding preserves global structure
    better than PCA, a score equal to 1 indicates equal performance to PCA,
    and a score below 1 indicates worse performance than PCA.
    """

    def evaluate(
        self, X: np.ndarray, coords: np.ndarray, ctx: EmbedContext | None = None
    ) -> EvalResult:
        gs = global_score(
            X,
            coords,
        )
        return EvalResult(
            eval_type="GS",
            label=str(self),
            info={},
            value=float(gs),
        )

    def __str__(self):
        return "Global Score Evaluation"


def global_loss(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute the global structure preservation error between X and Y.

    This function measures how well Y (after mean-centering) can
    linearly reconstruct X (also mean-centered). It finds the least
    squares linear transformation A mapping Y to X, then returns
    the mean squared error between X and A @ Y. A lower value
    indicates better preservation of global structure.

    Used in Trimap: https://github.com/eamid/trimap/blob/master/trimap/trimap_.py
    """
    X = X - np.mean(X, axis=0)
    Y = Y - np.mean(Y, axis=0)
    A = X.T @ (Y @ np.linalg.inv(Y.T @ Y))
    return np.mean(np.power(X.T - A @ Y.T, 2))


def global_score(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compare the global structure preservation of Y against a PCA baseline.

    The function performs PCA on X to obtain Y_pca (same dimension as Y),
    computes their respective global_loss values, and returns an
    exponential comparison score:

        exp( -(global_loss(X, Y) - global_loss(X, Y_pca)) / global_loss(X, Y_pca) )

    Scores above 1 indicate that Y preserves global structure better
    than PCA, scores equal to 1 indicate equal performance to PCA, and
    scores below 1 indicate worse performance than PCA.
    """
    n_dims = Y.shape[1]
    Y_pca = PCA(n_components=n_dims).fit_transform(X)
    gs_pca = global_loss(X, Y_pca)
    gs_emb = global_loss(X, Y)
    return np.exp(-(gs_emb - gs_pca) / gs_pca)
