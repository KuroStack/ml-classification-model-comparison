from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .base_model import BaseClassifier


class LogisticRegressionModel(BaseClassifier):
    """ logistic regression (softmax / cross-entropy) classifier."""

    def __init__(
        self,
        max_iter: int = 1000,
        C: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self._model = LogisticRegression(
            max_iter=max_iter,
            C=C,
            random_state=random_state,
            solver="lbfgs",
        )

    @property
    def model_name(self) -> str:
        return "Logistic Regression"

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)
