"""
clustering.py
==============
Unsupervised learning module: K-Means clustering with automatic k-selection
(silhouette analysis) and benchmarking against ground-truth labels using
standard external clustering metrics.

Typical usage
-------------
>>> from src.clustering import KMeansExplorer
>>> explorer = KMeansExplorer(k_range=range(2, 8))
>>> explorer.fit(X)
>>> explorer.plot_silhouette_curve()
>>> explorer.plot_best_clustering()
"""

from __future__ import annotations

from time import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class KMeansExplorer:
    """Fit K-Means over a range of ``k`` and select the best one by
    average silhouette score.

    Parameters
    ----------
    k_range : Iterable[int]
        Candidate numbers of clusters to try.
    n_init : int
        Number of K-Means restarts per k (passed to sklearn's KMeans).
    random_state : int
        Seed for reproducibility.
    """

    k_range: Iterable[int] = field(default_factory=lambda: range(2, 8))
    n_init: int = 10
    max_iter: int = 300
    random_state: int = 42

    def __post_init__(self):
        self.k_range = list(self.k_range)
        self.models_: dict[int, KMeans] = {}
        self.labels_by_k_: dict[int, np.ndarray] = {}
        self.silhouette_scores_: dict[int, float] = {}
        self.best_k_: Optional[int] = None
        self.X_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "KMeansExplorer":
        self.X_ = np.asarray(X)
        for k in self.k_range:
            km = KMeans(
                n_clusters=k,
                n_init=self.n_init,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            labels = km.fit_predict(self.X_)
            self.models_[k] = km
            self.labels_by_k_[k] = labels
            self.silhouette_scores_[k] = silhouette_score(self.X_, labels)

        self.best_k_ = max(self.silhouette_scores_, key=self.silhouette_scores_.get)
        return self

    @property
    def best_model(self) -> KMeans:
        self._check_fitted()
        return self.models_[self.best_k_]

    @property
    def best_labels(self) -> np.ndarray:
        self._check_fitted()
        return self.labels_by_k_[self.best_k_]

    def _check_fitted(self):
        if self.best_k_ is None:
            raise RuntimeError("Call .fit(X) before accessing results.")

    def summary(self) -> str:
        self._check_fitted()
        lines = [f"Best k by average silhouette: {self.best_k_} "
                 f"(score={self.silhouette_scores_[self.best_k_]:.3f})"]
        for k in self.k_range:
            lines.append(f"  k={k}: silhouette={self.silhouette_scores_[k]:.3f}")
        return "\n".join(lines)

    def plot_silhouette_curve(self, ax=None):
        self._check_fitted()
        ax = ax or plt.gca()
        ks = self.k_range
        scores = [self.silhouette_scores_[k] for k in ks]
        ax.plot(ks, scores, marker="o")
        ax.axvline(self.best_k_, color="grey", linestyle="--", alpha=0.6)
        ax.set_xlabel("Number of clusters k")
        ax.set_ylabel("Average silhouette score")
        ax.set_title("Silhouette score vs. k")
        ax.grid(True, linestyle="--", alpha=0.4)
        return ax

    def plot_best_clustering(self, ax=None):
        """Only meaningful for 2D feature spaces."""
        self._check_fitted()
        ax = ax or plt.gca()
        X, labels, model = self.X_, self.best_labels, self.best_model
        ax.scatter(X[:, 0], X[:, 1], s=15, c=labels, cmap="viridis")
        centers = model.cluster_centers_
        ax.scatter(centers[:, 0], centers[:, 1], s=200, marker="X", edgecolor="k")
        ax.set_title(f"K-Means (k={self.best_k_}), "
                     f"avg silhouette={self.silhouette_scores_[self.best_k_]:.3f}")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        return ax


def bench_k_means(kmeans: KMeans, name: str, data: np.ndarray, labels: np.ndarray) -> dict:
    """Benchmark a K-Means estimator against ground-truth labels.

    Scales the data with ``StandardScaler`` inside a pipeline, fits, and
    reports fit time plus inertia and five external clustering metrics
    (homogeneity, completeness, V-measure, ARI, AMI) alongside the
    silhouette score.

    Returns a dict so results can be collected into a pandas DataFrame,
    e.g. ``pd.DataFrame([bench_k_means(...) for ...])``.
    """
    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    pred_labels = estimator[-1].labels_

    result = {
        "name": name,
        "fit_time_s": fit_time,
        "inertia": estimator[-1].inertia_,
        "homogeneity": metrics.homogeneity_score(labels, pred_labels),
        "completeness": metrics.completeness_score(labels, pred_labels),
        "v_measure": metrics.v_measure_score(labels, pred_labels),
        "adjusted_rand": metrics.adjusted_rand_score(labels, pred_labels),
        "adjusted_mutual_info": metrics.adjusted_mutual_info_score(labels, pred_labels),
        "silhouette": metrics.silhouette_score(
            data, pred_labels, metric="euclidean",
            sample_size=min(300, len(data)),
        ),
    }
    return result
