from __future__ import annotations

import math

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from .base_model import BaseClassifier


class KNNModel(BaseClassifier):
    """k-NN classifier with Euclidean distance and K = sqrt(N) if K is not set."""

    def __init__(
        self,
        n_neighbors: int | None = None,
        metric: str = "euclidean",
        weights: str = "uniform",
    ) -> None:
        self._n_neighbors_init = n_neighbors
        self._metric = metric
        self._weights = weights
        self._model: KNeighborsClassifier | None = None

    @property
    def model_name(self) -> str:
        return "K-Nearest Neighbor"

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        n_train = X_train.shape[0]
        k = (
            self._n_neighbors_init
            if self._n_neighbors_init is not None
            else max(1, int(math.sqrt(n_train)))
        )
        if k % 2 == 0:
            k += 1

        self._model = KNeighborsClassifier(
            n_neighbors=k,
            metric=self._metric,
            weights=self._weights,
        )
        self._model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    @property
    def k_value(self) -> int | None:
        if self._model is None:
            return None
        return self._model.n_neighbors
