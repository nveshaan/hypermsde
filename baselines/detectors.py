"""Standard anomaly detectors on stored image embeddings, for the encoder comparison (scripts/run_detector.py).

Every detector is ``detector(X, support=None, **params) -> score`` with ``X`` the (n, d) evaluated embeddings and
``support`` the (K, d) normal support embeddings (few-shot) or None (zero-shot). Higher score = more anomalous.
Zero-shot is transductive, like HyperMSDE: the detector is fit on ``X`` itself, every point leaving itself out.
Few-shot fits on the support only (inductive).
"""
import numpy as np
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors


def cosine_text(X, support=None, text=None, temperature=0.01):
    """Softmax probability of the abnormal prompts over cosine similarity to the averaged normal and abnormal
    prompt embeddings (``text = (normal, abnormal)``, each (P, d)). Uses no support."""
    anchors = np.stack([text[0].mean(0), text[1].mean(0)])
    anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return softmax(X @ anchors.T / temperature, axis=1)[:, 1]


def knn(X, support=None, k=10, metric="cosine"):
    """Mean distance to the k nearest neighbours (in the support, or in X without the point itself)."""
    if support is None:
        dist, _ = NearestNeighbors(n_neighbors=min(k, len(X) - 1) + 1, metric=metric).fit(X).kneighbors(X)
        return dist[:, 1:].mean(1)
    dist, _ = NearestNeighbors(n_neighbors=min(k, len(support)), metric=metric).fit(support).kneighbors(X)
    return dist.mean(1)


def lof(X, support=None, k=20, metric="cosine"):
    """Local outlier factor. With a single support point LOF is undefined; it falls back to the distance to it."""
    if support is None:
        model = LocalOutlierFactor(n_neighbors=min(k, len(X) - 1), metric=metric).fit(X)
        return -model.negative_outlier_factor_
    if len(support) < 2:
        return knn(X, support, k=1, metric=metric)
    model = LocalOutlierFactor(n_neighbors=min(k, len(support) - 1), metric=metric, novelty=True).fit(support)
    return -model.score_samples(X)


def pca(X, support=None, variance=0.9):
    """Squared reconstruction error from the principal subspace explaining `variance` of the variance. Fit on the
    support, at most K - 1 components; a single support point leaves only its mean (distance to it)."""
    fit = X if support is None else support
    if len(fit) < 2:
        return ((X - fit.mean(0)) ** 2).sum(1)
    model = PCA(n_components=variance, svd_solver="full").fit(fit)
    return ((X - model.inverse_transform(model.transform(X))) ** 2).sum(1)
