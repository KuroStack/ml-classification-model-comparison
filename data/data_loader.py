from __future__ import annotations

import ssl
import urllib.request
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

_TARGET_COL = "fetal_health"

_FEATURE_COLS = [
    "baseline_value",
    "accelerations",
    "fetal_movement",
    "uterine_contractions",
    "light_decelerations",
    "severe_decelerations",
    "prolongued_decelerations",
    "abnormal_short_term_variability",
    "mean_value_of_short_term_variability",
    "percentage_of_time_with_abnormal_long_term_variability",
    "mean_value_of_long_term_variability",
    "histogram_width",
    "histogram_min",
    "histogram_max",
    "histogram_number_of_peaks",
    "histogram_number_of_zeroes",
    "histogram_mode",
    "histogram_mean",
    "histogram_median",
    "histogram_variance",
    "histogram_tendency",
]

_LABEL_MAP = {
    1: "Normal",
    2: "Suspect",
    3: "Pathological",
    1.0: "Normal",
    2.0: "Suspect",
    3.0: "Pathological",
    "1": "Normal",
    "2": "Suspect",
    "3": "Pathological",
    "1.0": "Normal",
    "2.0": "Suspect",
    "3.0": "Pathological",
    "normal": "Normal",
    "suspect": "Suspect",
    "pathological": "Pathological",
    "pathologic": "Pathological",
}


def _allow_unverified_https() -> None:
    ctx = ssl._create_unverified_context()
    orig = urllib.request.urlopen

    def _urlopen(url, data=None, timeout=60, *args, **kwargs):
        kwargs["context"] = ctx
        return orig(url, data, timeout, *args, **kwargs)

    urllib.request.urlopen = _urlopen


class FetalHealthDataLoader:
    """Loads UCI Cardiotocography (id=193) via ucimlrepo and prepares train/test splits."""

    def __init__(self, test_size: float = 0.20, random_state: int = 42) -> None:
        self._test_size = test_size
        self._random_state = random_state
        self._scaler: StandardScaler | None = None
        self._target_encoder: LabelEncoder = LabelEncoder()
        self._feature_names: list[str] = []
        self._is_loaded: bool = False
        self._impute_medians: pd.Series | None = None

    def load(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        df = self._clean(self._download())
        y = self._target_encoder.fit_transform(df[_TARGET_COL].astype(str))
        X_df = df.drop(columns=[_TARGET_COL])
        self._feature_names = X_df.columns.tolist()
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X_df, y,
            test_size=self._test_size,
            random_state=self._random_state,
            stratify=y,
        )
        self._impute_medians = X_train_df.median(numeric_only=True)
        X_train_df = X_train_df.fillna(self._impute_medians)
        X_test_df = X_test_df.fillna(self._impute_medians)
        self._scaler = StandardScaler()
        X_train = self._scaler.fit_transform(X_train_df.values)
        X_test = self._scaler.transform(X_test_df.values)
        self._is_loaded = True
        return X_train, X_test, y_train.astype(int), y_test.astype(int), self._feature_names

    def preprocess_uploaded(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self._is_loaded:
            raise RuntimeError("Call load() before preprocess_uploaded().")

        df = self._normalise_columns(df.copy())
        has_target = _TARGET_COL in df.columns
        target_raw = df[_TARGET_COL] if has_target else None
        df = df.drop(columns=[_TARGET_COL], errors="ignore")

        for col in self._feature_names:
            if col not in df.columns:
                df[col] = np.nan
        df = df[self._feature_names].apply(pd.to_numeric, errors="coerce")
        df = df.fillna(self._impute_medians)

        X = self._scaler.transform(df.values)

        y = np.array([])
        if has_target:
            mapped = target_raw.map(self._map_label)
            mapped = mapped.fillna(self._target_encoder.classes_[0])
            y = self._target_encoder.transform(mapped.astype(str))
        return X, y

    @property
    def class_names(self) -> list[str]:
        return list(self._target_encoder.classes_)

    def _map_label(self, v) -> str:
        if pd.isna(v):
            return "Normal"
        if v in _LABEL_MAP:
            return _LABEL_MAP[v]
        key = str(v).strip().lower()
        return _LABEL_MAP.get(key, str(v).strip().title())

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [
            c.strip().lower().replace(" ", "_").replace("-", "_")
            for c in df.columns
        ]
        rename = {
            "lb": "baseline_value",
            "ac": "accelerations",
            "fm": "fetal_movement",
            "uc": "uterine_contractions",
            "dl": "light_decelerations",
            "ds": "severe_decelerations",
            "dp": "prolongued_decelerations",
            "astv": "abnormal_short_term_variability",
            "mstv": "mean_value_of_short_term_variability",
            "altv": "percentage_of_time_with_abnormal_long_term_variability",
            "mltv": "mean_value_of_long_term_variability",
            "width": "histogram_width",
            "min": "histogram_min",
            "max": "histogram_max",
            "nmax": "histogram_number_of_peaks",
            "nzeros": "histogram_number_of_zeroes",
            "mode": "histogram_mode",
            "mean": "histogram_mean",
            "median": "histogram_median",
            "variance": "histogram_variance",
            "tendency": "histogram_tendency",
            "nsp": _TARGET_COL,
            "prolonged_decelerations": "prolongued_decelerations",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        df = df.drop(columns=["class"], errors="ignore")
        return df

    def _fetch_from_uci(self) -> pd.DataFrame:
        repo = fetch_ucirepo(id=193)
        X = repo.data.features.copy()
        y = repo.data.targets.copy()
        if "NSP" not in y.columns:
            raise ValueError("UCI Cardiotocography response has no NSP target.")
        df = X.copy()
        df[_TARGET_COL] = y["NSP"]
        return df

    def _download(self) -> pd.DataFrame:
        try:
            return self._fetch_from_uci()
        except Exception:
            _allow_unverified_https()
            return self._fetch_from_uci()

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._normalise_columns(df.copy())
        if _TARGET_COL not in df.columns:
            raise ValueError("Dataset has no fetal_health / NSP target column.")

        available = [c for c in _FEATURE_COLS if c in df.columns]
        if len(available) < 12:
            numeric = df.select_dtypes(include=["number"]).columns.tolist()
            available = [c for c in numeric if c != _TARGET_COL]

        df = df[available + [_TARGET_COL]]
        for col in available:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[_TARGET_COL] = df[_TARGET_COL].map(self._map_label)
        df = df.dropna(subset=[_TARGET_COL]).reset_index(drop=True)
        self._feature_names = available
        return df
