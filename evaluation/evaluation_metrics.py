from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class MetricsResult:
    """Holds Accuracy, AUC, Precision, Recall, F1, MCC, confusion matrix, and report."""

    model_name: str
    accuracy: float
    auc: float
    precision: float
    recall: float
    f1: float
    mcc: float
    confusion_mat: np.ndarray
    classification_rep: str

    def to_dict(self) -> Dict[str, float]:
        return {
            "Accuracy": self.accuracy,
            "AUC": self.auc,
            "Precision": self.precision,
            "Recall": self.recall,
            "F1 Score": self.f1,
            "MCC": self.mcc,
        }


class EvaluationMetrics:
    """Computes the assignment metrics for a classifier's predictions."""

    def __init__(self, n_classes: int = 2) -> None:
        self._is_binary: bool = n_classes == 2

    def compute(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        class_names: list[str] | None = None,
    ) -> MetricsResult:
        accuracy = round(float(accuracy_score(y_true, y_pred)), 4)

        if self._is_binary:
            auc = round(float(roc_auc_score(y_true, y_prob[:, 1])), 4)
            avg = "binary"
        else:
            auc = round(
                float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")),
                4,
            )
            avg = "weighted"

        precision = round(
            float(precision_score(y_true, y_pred, average=avg, zero_division=0)), 4
        )
        recall = round(
            float(recall_score(y_true, y_pred, average=avg, zero_division=0)), 4
        )
        f1 = round(float(f1_score(y_true, y_pred, average=avg, zero_division=0)), 4)
        mcc = round(float(matthews_corrcoef(y_true, y_pred)), 4)

        return MetricsResult(
            model_name=model_name,
            accuracy=accuracy,
            auc=auc,
            precision=precision,
            recall=recall,
            f1=f1,
            mcc=mcc,
            confusion_mat=confusion_matrix(y_true, y_pred),
            classification_rep=classification_report(
                y_true, y_pred, zero_division=0, target_names=class_names
            ),
        )
