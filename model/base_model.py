from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseClassifier(ABC):
    """Abstract interface for train, predict, and predict_proba."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...
