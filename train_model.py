"""
Train an XGBoost classifier to filter rules-based trading signals.

Reads:  training_data/{instrument}_signals.csv
Writes: models/{instrument}_model.joblib
        models/{instrument}_threshold.txt

The training uses a CHRONOLOGICAL train/validation split — never random.
Random splits leak future market behavior into training and produce
backtests that fall apart in live trading.

Usage:
    python train_model.py
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

from config import INSTRUMENTS, TAKE_PROFIT_PCT, STOP_LOSS_PCT
from features import FEATURE_COLUMNS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# Asymmetric payoff: a winning trade gains 1.5%, a losing trade loses 0.8%.
# When picking a threshold we want to MAXIMISE expected P&L, not accuracy.
WIN_RETURN = TAKE_PROFIT_PCT  # +0.015
LOSS_RETURN = -STOP_LOSS_PCT  # -0.008

TRAIN_FRACTION = 0.75  # first 75% of data → train, last 25% → validate
MIN_TRADES_PER_THRESHOLD = 5  # relaxed for early training with small validation sets


def expected_pnl(y_true, y_pred_proba, threshold):
    """
    Expected total P&L if we take only trades where model confidence >= threshold.

    Returns (expected_pnl_pct, num_trades_taken, win_rate_of_taken_trades).
    """
    taken_mask = y_pred_proba >= threshold
    n_taken = int(taken_mask.sum())
    if n_taken < MIN_TRADES_PER_THRESHOLD:
        return -np.inf, n_taken, 0.0

    actual_outcomes = y_true[taken_mask]
    wins = int(actual_outcomes.sum())
    losses = n_taken - wins
    pnl = wins * WIN_RETURN + losses * LOSS_RETURN
    return pnl, n_taken, wins / n_taken


def find_best_threshold(y_true, y_pred_proba):
    """Sweep thresholds from 0.50 → 0.85 and pick the one with highest P&L."""
    best = {"threshold": 0.5, "pnl": -np.inf, "n_trades": 0, "win_rate": 0.0}
    for t in np.arange(0.50, 0.86, 0.01):
        pnl, n, wr = expected_pnl(y_true, y_pred_proba, t)
        if pnl > best["pnl"]:
            best = {
                "threshold": round(float(t), 2),
                "pnl": pnl,
                "n_trades": n,
                "win_rate": wr,
            }
    return best


def train_one_instrument(csv_path: Path, out_dir: Path, instrument: str):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"]).sort_values("timestamp")
    source = "backtest+live" if "live" in str(csv_path) else "backtest only"
    logger.info(f"\n━━━ {instrument} ━━━")
    logger.info(
        f"Source: {source} | {len(df)} labeled signals | baseline win rate {df['y'].mean():.1%}"
    )

    if len(df) < 30:
        logger.warning(
            f"{instrument}: Only {len(df)} samples — need 30+ to train. Skipping."
        )
        return

    # Chronological split
    split_idx = int(len(df) * TRAIN_FRACTION)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["y"]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df["y"]

    logger.info(f"Train: {len(X_train)} signals  |  Validation: {len(X_val)} signals")

    # Conservative hyperparameters — light regularisation to resist overfitting
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ── Validation diagnostics ────────────────────────────────────────────
    val_proba = model.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_val, val_proba)
    logger.info(f"\nValidation AUC: {auc:.3f}  (0.5 = random, 0.7+ = useful)")
    logger.info(
        "\n"
        + classification_report(y_val, val_pred, target_names=["Loss (0)", "Win (1)"])
    )

    # ── Threshold selection on validation set ─────────────────────────────
    best = find_best_threshold(y_val.values, val_proba)
    baseline_pnl = y_val.sum() * WIN_RETURN + (len(y_val) - y_val.sum()) * LOSS_RETURN

    logger.info("Threshold selection on validation set:")
    logger.info(
        f"  No filter (take all): {len(y_val)} trades, "
        f"{y_val.mean():.1%} WR, P&L = {baseline_pnl:+.2%}"
    )
    logger.info(
        f"  Best threshold:       {best['threshold']:.2f} "
        f"→ {best['n_trades']} trades, {best['win_rate']:.1%} WR, "
        f"P&L = {best['pnl']:+.2%}"
    )

    # ── Feature importance ────────────────────────────────────────────────
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    importance = importance.sort_values(ascending=False)
    logger.info("\nTop features:\n" + importance.head(8).to_string())

    # ── Save artefacts ────────────────────────────────────────────────────
    model_path = out_dir / f"{instrument}_model.joblib"
    thresh_path = out_dir / f"{instrument}_threshold.txt"
    joblib.dump(model, model_path)
    thresh_path.write_text(str(best["threshold"]))
    logger.info(f"\nSaved model    → {model_path}")
    logger.info(f"Saved threshold → {thresh_path} ({best['threshold']})")


if __name__ == "__main__":
    data_dir = Path("training_data")
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)

    for instrument in INSTRUMENTS:
        csv_path = data_dir / f"{instrument}_signals.csv"
        if not csv_path.exists():
            logger.error(f"Missing {csv_path}. Run `python backtest.py` first.")
            continue
        train_one_instrument(csv_path, out_dir, instrument)
