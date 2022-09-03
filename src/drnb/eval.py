import abc
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import scipy.stats
from sklearn.decomposition import PCA

from drnb.embed import get_coords
from drnb.neighbors import get_neighbors


class EmbeddingEval(abc.ABC):
    def evaluate(self, X, coords):
        pass


@dataclass
class RandomTripletEval(EmbeddingEval):
    random_state: int = None
    n_triplets_per_point: int = 5

    def evaluate(self, X, coords):
        rte = random_triplet_eval(
            X,
            coords,
            random_state=self.random_state,
            n_triplets_per_point=self.n_triplets_per_point,
        )
        return ("rte", rte)


def create_evaluators(eval_metrics=None):
    if eval_metrics is None:
        return []

    if not isinstance(eval_metrics, Iterable):
        eval_metrics = [eval_metrics]

    evaluators = []
    for embed_eval in eval_metrics:
        if isinstance(embed_eval, tuple):
            if len(embed_eval) != 2:
                raise ValueError("Bad format for eval spec")
            embed_eval_name = embed_eval[0]
            eval_kwds = embed_eval[1]
        else:
            embed_eval_name = embed_eval
            eval_kwds = {}

        embed_eval_name = embed_eval_name.lower()
        if embed_eval_name == "gs":
            ctor = GlobalScore
        elif embed_eval_name == "rte":
            ctor = RandomTripletEval
        elif embed_eval_name == "rpc":
            ctor = RandomPairCorrelEval
        elif embed_eval_name == "nnp":
            ctor = NbrPreservationEval
        else:
            raise ValueError(f"Unknown embed eval option '{embed_eval_name}'")
        evaluators.append(ctor(**eval_kwds))

    return evaluators


def evaluate_embedding(evaluators, X, embedding):
    coords = get_coords(embedding)
    return [evaluator.evaluate(X, coords) for evaluator in evaluators]


# https://github.com/YingfanWang/PaCMAP/blob/c7c45dbd0fec7736764d0e28203eb0e0515f3427/evaluation/evaluation.py
def random_triplet_eval(
    X, X_new, triplets=None, random_state=None, n_triplets_per_point=5
):
    n_obs = X.shape[0]

    # Sampling Triplets
    # Five triplet per point
    if triplets is None:
        triplets = get_triplets(
            X, seed=random_state, n_triplets_per_point=n_triplets_per_point
        )
    else:
        validate_triplets(triplets, n_obs)
        n_triplets_per_point = triplets.shape[1]

    # 3D (Nx1x1) array where e.g. anchors[i][0][0] = i: [[[0]], [[1]], [[2]] ... [[n_obs]]]
    anchors = np.arange(n_obs).reshape((-1, 1, 1))

    # Calculate the distances and generate labels
    # broadcasting to b flattens the anchors and triplets to generate a list of pairs:
    # [0, triplet00], [0, triplet01], [0, triplet10], [0, triplet11] etc.
    b = np.broadcast(anchors, triplets)
    labels = calc_labels(X, b)

    # Calculate distances for LD
    b = np.broadcast(anchors, triplets)
    pred_vals = calc_labels(X_new, b)

    correct = np.sum(pred_vals == labels)
    acc = correct / n_obs / n_triplets_per_point
    return acc


def get_triplets(X, seed=None, n_triplets_per_point=5):
    anchors = np.arange(X.shape[0])
    rng = np.random.default_rng(seed=seed)
    # for each row of X generate n_triplets_per_point pairs sampled from anchors
    triplets = rng.choice(anchors, (X.shape[0], n_triplets_per_point, 2))
    return triplets


def calc_distances(X, pairs):
    distances = np.empty(pairs.shape)
    distances.flat = [np.linalg.norm(X[u] - X[v]) for (u, v) in pairs]
    return distances


def calc_labels(X, pairs):
    distances = calc_distances(X, pairs)
    return distances[:, :, 0] < distances[:, :, 1]


def validate_triplets(triplets, n_obs):
    if len(triplets.shape) != 3 or triplets.shape[2] != 2 or triplets.shape[0] != n_obs:
        raise ValueError(
            f"triplets should have shape ({n_obs}, n_triplets_per_point, 2)"
        )


def triplets_to_pairs(triplets, n_obs):
    anchors = np.arange(n_obs).reshape((-1, 1, 1))
    return np.broadcast(anchors, triplets)


def random_pair_correl_eval(
    X, X_new, triplets=None, random_state=None, n_triplets_per_point=5
):
    n_obs = X.shape[0]
    if triplets is None:
        triplets = get_triplets(
            X, seed=random_state, n_triplets_per_point=n_triplets_per_point
        )
    else:
        validate_triplets(triplets, n_obs)
        n_triplets_per_point = triplets.shape[1]

    anchors = np.arange(n_obs).reshape((-1, 1, 1))
    bpairs = np.broadcast(anchors, triplets)
    d_X = calc_distances(X, bpairs)

    bpairs = np.broadcast(anchors, triplets)
    d_Xnew = calc_distances(X_new, bpairs)

    return scipy.stats.pearsonr(d_X.flatten(), d_Xnew.flatten()).statistic


@dataclass
class RandomPairCorrelEval(EmbeddingEval):
    random_state: int = None
    n_triplets_per_point: int = 5

    def evaluate(self, X, coords):
        rpc = random_pair_correl_eval(
            X,
            coords,
            random_state=self.random_state,
            n_triplets_per_point=self.n_triplets_per_point,
        )
        return ("rpc", rpc)


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


def nn_accv(approx_indices, true_indices):
    result = np.zeros(approx_indices.shape[0])
    for i in range(approx_indices.shape[0]):
        n_correct = np.intersect1d(approx_indices[i], true_indices[i]).shape[0]
        result[i] = n_correct / true_indices.shape[1]
    return result


def nn_acc(approx_indices, true_indices):
    return np.mean(nn_accv(approx_indices, true_indices))


def nbr_pres(
    X,
    Y,
    n_nbrs=15,
    x_method="exact",
    x_metric="euclidean",
    x_method_kwds=None,
    y_method="exact",
    y_metric="euclidean",
    y_method_kwds=None,
    verbose=False,
):

    Xnbrs = get_neighbors(
        X,
        metric=x_metric,
        n_neighbors=n_nbrs,
        method=x_method,
        return_distance=False,
        method_kwds=x_method_kwds,
        verbose=verbose,
    )
    Ynbrs = get_neighbors(
        Y,
        metric=y_metric,
        n_neighbors=n_nbrs,
        method=y_method,
        return_distance=False,
        method_kwds=y_method_kwds,
        verbose=verbose,
    )

    return nn_acc(Ynbrs, Xnbrs)


@dataclass
class NbrPreservationEval(EmbeddingEval):
    n_neighbors: int = 15
    verbose: bool = False

    def evaluate(self, X, coords):
        nnp = nbr_pres(X, coords, n_nbrs=self.n_neighbors, verbose=self.verbose)
        return (f"nnp{self.n_neighbors}", nnp)
