from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from .base import EmbeddingEval


@dataclass
class GlobalScore(EmbeddingEval):
    def evaluate(self, X, coords):
        gs = global_score(
            X,
            coords,
        )
        return ("gs", gs)


# Used in Trimap: https://github.com/eamid/trimap/blob/master/trimap/trimap_.py
def global_loss(X, Y):
    X = X - np.mean(X, axis=0)
    Y = Y - np.mean(Y, axis=0)
    A = X.T @ (Y @ np.linalg.inv(Y.T @ Y))
    return np.mean(np.power(X.T - A @ Y.T, 2))


def global_score(X, Y):
    n_dims = Y.shape[1]
    Y_pca = PCA(n_components=n_dims).fit_transform(X)
    gs_pca = global_loss(X, Y_pca)
    gs_emb = global_loss(X, Y)
    return np.exp(-(gs_emb - gs_pca) / gs_pca)
