"""
Live confidence filter — loads the trained XGBoost model and threshold,
then approves or skips incoming signals based on predicted win probability.

If the model file is missing, every signal passes through (transparent fallback).
This means the bot still works in pure rules mode before training is complete.
"""
import logging
from pathlib import Path

import joblib
import pandas as pd

from features import compute_features, FEATURE_COLUMNS

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")


class MLFilter:
    def __init__(self, instrument: str):
        self.instrument = instrument
        self.model      = None
        self.threshold  = 0.5
        self._load()

    def _load(self):
        model_path  = MODEL_DIR / f"{self.instrument}_model.joblib"
        thresh_path = MODEL_DIR / f"{self.instrument}_threshold.txt"

        if not model_path.exists() or not thresh_path.exists():
            logger.warning(
                f"{self.instrument}: no trained model found at {model_path} — "
                "ML filter disabled, all signals will pass through."
            )
            return

        self.model     = joblib.load(model_path)
        self.threshold = float(thresh_path.read_text().strip())
        logger.info(
            f"{self.instrument}: loaded ML filter "
            f"(threshold = {self.threshold:.2f})"
        )

    @property
    def is_active(self) -> bool:
        return self.model is not None

    def predict_proba(self, df: pd.DataFrame, signal: str) -> float | None:
        """Return P(win) for the given signal, or None if features unavailable."""
        if not self.is_active:
            return None

        features = compute_features(df, signal)
        if features is None:
            return None

        x = pd.DataFrame([{k: features[k] for k in FEATURE_COLUMNS}])
        return float(self.model.predict_proba(x)[0, 1])

    def approve(self, df: pd.DataFrame, signal: str) -> tuple[bool, float | None]:
        """
        Returns (allow_trade, predicted_probability).
        If filter is disabled, returns (True, None) — let everything through.
        """
        if not self.is_active:
            return True, None

        p = self.predict_proba(df, signal)
        if p is None:
            logger.warning(f"{self.instrument}: features unavailable — defaulting to approve.")
            return True, None

        approved = p >= self.threshold
        return approved, p
