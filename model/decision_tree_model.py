from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .base_model import BaseClassifier


class DecisionTreeModel(BaseClassifier):
    """Decision tree classifier using entropy (information gain)."""

    def __init__(
        self,
        criterion: str = "entropy",
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        random_state: int = 42,
    ) -> None:
        self._model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    @property
    def model_name(self) -> str:
        return "Decision Tree"

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)
