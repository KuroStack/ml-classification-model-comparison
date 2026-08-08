from __future__ import annotations

import numpy as np
from sklearn.naive_bayes import GaussianNB

from .base_model import BaseClassifier


class NaiveBayesModel(BaseClassifier):
    """Gaussian Naive Bayes classifier."""

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        self._model = GaussianNB(var_smoothing=var_smoothing)

    @property
    def model_name(self) -> str:
        return "Naive Bayes (Gaussian)"

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)
