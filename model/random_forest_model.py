from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .base_model import BaseClassifier


class RandomForestModel(BaseClassifier):
    """Random forest ensemble of entropy decision trees (bagging)."""

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "entropy",
        max_features: str = "sqrt",
        max_depth: int | None = None,
        random_state: int = 42,
    ) -> None:
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_features=max_features,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

    @property
    def model_name(self) -> str:
        return "Random Forest (Ensemble)"

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)
