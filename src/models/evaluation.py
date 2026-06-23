"""
evaluation.py
-------------
Walk-forward out-of-sample evaluation harness and loss functions.

WHY WALK-FORWARD (not random train/test split):
  Financial time series have temporal dependence — a random split would leak
  future information into training and produce optimistic estimates. Walk-forward
  re-fits the model on all data up to time t and evaluates on the next window,
  rolling forward until the end of the sample. This is the only valid evaluation
  framework for time-series forecasting.

LOSS FUNCTIONS:
  - QLIKE: the standard proper loss for volatility forecasting.
      QLIKE = mean( RV/RV_hat - log(RV/RV_hat) - 1 )
      It is asymmetric (penalises under-prediction of variance more than
      over-prediction) and is scale-invariant. Lower is better.
  - MSE on log-RV: mean( (log RV - log RV_hat)^2 ).
      Easier to interpret; also standard in the literature.

REGIME BREAKDOWN:
  We separately report performance during the turbulent sub-periods defined
  in config.yaml — this is the key test of whether TDA adds value specifically
  during stress periods, as Souto (2023) found on daily equity data.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import yaml


# ── Loss functions ────────────────────────────────────────────────────────────

def qlike(rv_true: np.ndarray, rv_pred: np.ndarray) -> float:
    """QLIKE loss. Lower is better. Clips predictions to avoid log(0)."""
    rv_pred = np.clip(rv_pred, 1e-20, None)
    rv_true = np.clip(rv_true, 1e-20, None)
    return float(np.mean(rv_true / rv_pred - np.log(rv_true / rv_pred) - 1))


def mse_log_rv(rv_true: np.ndarray, rv_pred: np.ndarray) -> float:
    """MSE on log-RV. Lower is better."""
    rv_pred = np.clip(rv_pred, 1e-20, None)
    rv_true = np.clip(rv_true, 1e-20, None)
    return float(np.mean((np.log(rv_true) - np.log(rv_pred)) ** 2))


# ── Model protocol (duck-typed interface) ────────────────────────────────────

class Forecaster(Protocol):
    def fit(self, df: pd.DataFrame) -> "Forecaster": ...
    def predict(self, df: pd.DataFrame) -> np.ndarray: ...


# ── Walk-forward result container ────────────────────────────────────────────

@dataclass
class WalkForwardResult:
    model_name:  str
    symbol:      str
    predictions: pd.Series          # index = timestamp, values = predicted RV
    actuals:     pd.Series          # index = timestamp, values = actual RV
    fold_metrics: list[dict] = field(default_factory=list)

    def overall_metrics(self) -> dict:
        rv_t = self.actuals.values
        rv_p = self.predictions.values
        return {
            "model":   self.model_name,
            "symbol":  self.symbol,
            "n_obs":   len(rv_t),
            "qlike":   qlike(rv_t, rv_p),
            "mse_log": mse_log_rv(rv_t, rv_p),
        }

    def regime_metrics(self, turbulent_periods: list[dict]) -> list[dict]:
        """Compute metrics separately for each turbulent sub-period."""
        rows = []
        for period in turbulent_periods:
            mask = (
                (self.actuals.index >= pd.Timestamp(period["start"], tz="UTC")) &
                (self.actuals.index <= pd.Timestamp(period["end"],   tz="UTC"))
            )
            if mask.sum() < 10:
                continue
            rv_t = self.actuals[mask].values
            rv_p = self.predictions[mask].values
            rows.append({
                "model":   self.model_name,
                "symbol":  self.symbol,
                "period":  period["name"],
                "n_obs":   len(rv_t),
                "qlike":   qlike(rv_t, rv_p),
                "mse_log": mse_log_rv(rv_t, rv_p),
            })
        return rows


# ── Walk-forward engine ───────────────────────────────────────────────────────

def walk_forward(
    model_factory,          # callable() → fresh Forecaster instance
    model_name: str,
    symbol: str,
    df: pd.DataFrame,       # full feature DataFrame (train + OOS)
    oos_start: str,
    step_days: int = 30,    # re-fit every N days
    min_train_rows: int = 5000,
) -> WalkForwardResult:
    """
    Expanding-window walk-forward evaluation.

    At each step:
      1. Train on all data before the current window start.
      2. Predict on the next `step_days` days.
      3. Advance the window.

    No data from the test window is ever used during fitting.
    """
    oos_start_ts = pd.Timestamp(oos_start, tz="UTC")
    step = pd.Timedelta(days=step_days)

    all_preds  = {}
    all_actual = {}
    fold_metrics = []

    window_start = oos_start_ts
    window_end   = window_start + step

    total_end = df.index.max()

    print(f"  Walk-forward [{model_name} | {symbol}]: "
          f"OOS {oos_start} → {total_end.date()}, re-fit every {step_days}d")

    fold = 0
    while window_start < total_end:
        train_df = df[df.index < window_start]
        test_df  = df[(df.index >= window_start) & (df.index < window_end)]

        if len(train_df) < min_train_rows or len(test_df) == 0:
            window_start = window_end
            window_end   = window_start + step
            continue

        model = model_factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_df)

        preds = model.predict(test_df)

        for ts, p, a in zip(test_df.index, preds, test_df["rv_target"].values):
            all_preds[ts]  = p
            all_actual[ts] = a

        rv_t = test_df["rv_target"].values
        fold_metrics.append({
            "fold": fold,
            "start": window_start.date(),
            "end":   min(window_end, total_end).date(),
            "n":     len(test_df),
            "qlike":   qlike(rv_t, preds),
            "mse_log": mse_log_rv(rv_t, preds),
        })

        fold += 1
        window_start = window_end
        window_end   = window_start + step

    predictions = pd.Series(all_preds,  name="predicted_rv").sort_index()
    actuals     = pd.Series(all_actual, name="actual_rv").sort_index()

    return WalkForwardResult(
        model_name=model_name,
        symbol=symbol,
        predictions=predictions,
        actuals=actuals,
        fold_metrics=fold_metrics,
    )


# ── Runner ────────────────────────────────────────────────────────────────────

def run_baselines(cfg_path: str = "config.yaml",
                  results_dir: str = "results") -> pd.DataFrame:
    """
    Run HAR-RV and XGBoost baselines for all symbols, save results.
    Returns a summary DataFrame.
    """
    from src.models.har_rv import HARRV, HAR_FEATURES
    from src.models.xgboost_baseline import XGBBaseline

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    proc_dir   = Path(cfg["data"]["proc_dir"])
    out_dir    = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    oos_start       = cfg["evaluation"]["oos_start"]
    step_days       = cfg["evaluation"]["step_size"]
    turbulent       = cfg["turbulent_periods"]

    all_metrics  = []
    regime_rows  = []

    for symbol in cfg["data"]["symbols"]:
        safe = symbol.replace("/", "-")
        path = proc_dir / f"{safe}_rv_features.csv.gz"
        if not path.exists():
            print(f"Missing {path}, skipping.")
            continue

        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        print(f"\n=== {symbol}: {len(df):,} rows ===")

        models = {
            "HAR-RV":  lambda: HARRV(HAR_FEATURES),
            "XGBoost": lambda: XGBBaseline(HAR_FEATURES),
        }

        for name, factory in models.items():
            result = walk_forward(factory, name, symbol, df,
                                  oos_start=oos_start, step_days=step_days)

            # Save predictions
            pred_df = pd.DataFrame({
                "actual_rv":    result.actuals,
                "predicted_rv": result.predictions,
            })
            pred_df.to_csv(out_dir / f"{safe}_{name.lower().replace('-','_')}_preds.csv.gz", compression="gzip")

            # Metrics
            m = result.overall_metrics()
            all_metrics.append(m)
            regime_rows.extend(result.regime_metrics(turbulent))

            print(f"  {name}: QLIKE={m['qlike']:.6f}  MSE-log={m['mse_log']:.6f}  n={m['n_obs']:,}")

    summary = pd.DataFrame(all_metrics)
    summary.to_csv(out_dir / "baseline_summary.csv", index=False)

    if regime_rows:
        regime_df = pd.DataFrame(regime_rows)
        regime_df.to_csv(out_dir / "baseline_regime_breakdown.csv", index=False)
        print(f"\nRegime breakdown saved → {out_dir}/baseline_regime_breakdown.csv")

    print(f"\nBaseline summary saved → {out_dir}/baseline_summary.csv")
    return summary


if __name__ == "__main__":
    summary = run_baselines()
    print("\n=== Baseline Results ===")
    print(summary.to_string(index=False))
