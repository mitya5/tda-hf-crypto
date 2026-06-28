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
    purge_bars: int = 0,    # drop last N train rows (target look-ahead embargo)
) -> WalkForwardResult:
    """
    Expanding-window walk-forward evaluation.

    At each step:
      1. Train on all data before the current window start.
      2. Predict on the next `step_days` days.
      3. Advance the window.

    No data from the test window is ever used during fitting.

    PURGING: the RV target at time t looks forward `horizon` bars, so the last
    few training rows before the test window carry labels that overlap the test
    period — a subtle look-ahead leak. We drop the final `purge_bars` training
    rows so no training label peeks into the OOS window (cf. López de Prado's
    purged cross-validation).
    """
    oos_start_ts = pd.Timestamp(oos_start, tz="UTC")
    step = pd.Timedelta(days=step_days)

    pred_chunks  = []   # list of (index, pred-array) per fold
    actual_chunks = []
    fold_metrics = []

    window_start = oos_start_ts
    window_end   = window_start + step

    total_end = df.index.max()

    print(f"  Walk-forward [{model_name} | {symbol}]: "
          f"OOS {oos_start} → {total_end.date()}, re-fit every {step_days}d, purge {purge_bars} bars")

    fold = 0
    while window_start < total_end:
        train_df = df[df.index < window_start]
        if purge_bars > 0:
            train_df = train_df.iloc[:-purge_bars]
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
        rv_t  = test_df["rv_target"].values

        pred_chunks.append(pd.Series(preds, index=test_df.index))
        actual_chunks.append(test_df["rv_target"])

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

    predictions = (pd.concat(pred_chunks).sort_index().rename("predicted_rv")
                   if pred_chunks else pd.Series(dtype=float, name="predicted_rv"))
    actuals     = (pd.concat(actual_chunks).sort_index().rename("actual_rv")
                   if actual_chunks else pd.Series(dtype=float, name="actual_rv"))

    return WalkForwardResult(
        model_name=model_name,
        symbol=symbol,
        predictions=predictions,
        actuals=actuals,
        fold_metrics=fold_metrics,
    )


# ── Feature sets ──────────────────────────────────────────────────────────────

# Canonical TDA column names (kept in sync with src/features/tda.py). Hardcoded so
# the evaluation harness does not need to import ripser when only baselines are run.
TDA_FEATURES = [
    "tda_wass_h1", "tda_wass_h0", "tda_bottleneck_h1",
    "tda_pers_entropy_h0", "tda_pers_entropy_h1",
    "tda_max_pers_h1", "tda_total_pers_h1", "tda_total_pers_h0",
    "tda_n_loops_h1", "tda_landscape_l2_h0", "tda_landscape_l2_h1",
    "tda_degenerate",
]

# Parsimonious TDA additions for the LINEAR HAR model: a change signal plus the
# features most orthogonal to the RV lags (entropy / loop count). XGBoost can
# absorb the full set, so it uses all available TDA columns.
HAR_TDA_EXTRA = ["tda_wass_h1", "tda_pers_entropy_h1", "tda_n_loops_h1", "tda_max_pers_h1"]

# Non-topological 1-min realized-vol controls (see src/features/controls.py). These
# isolate the TOPOLOGY effect from the FREQUENCY effect: several TDA features are
# ~0.8 correlated with 1-min RV, which the 5-min baseline never sees. The defensible
# claim is "TDA helps *beyond* a 1-min RV control", tested by HAR+RV1 → HAR+RV1+TDA.
CONTROL_FEATURES = ["log_rv1_15", "log_rv1_60"]


# ── Runner ────────────────────────────────────────────────────────────────────

def run_evaluation(cfg_path: str = "config.yaml",
                   results_dir: str = "results") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run HAR-RV, XGBoost, and (if TDA feature files exist) HAR+TDA and XGBoost+TDA
    for all symbols, walk-forward OOS. Saves the summary, the per-regime breakdown,
    and Diebold–Mariano tests of each TDA model against its baseline.

    Returns (summary_df, dm_df).
    """
    from src.models.har_rv import HARRV, HAR_FEATURES
    from src.models.xgboost_baseline import XGBBaseline
    from src.models.significance import qlike_pointwise, diebold_mariano

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    proc_dir = Path(cfg["data"]["proc_dir"])
    out_dir  = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    oos_start = cfg["evaluation"]["oos_start"]
    step_days = cfg["evaluation"]["step_size"]
    turbulent = cfg["turbulent_periods"]

    # Embargo the forward-looking target horizon so training labels never overlap
    # the OOS window (horizon minutes / subsampling minutes = bars to purge).
    h_bars = (cfg["realized_vol"]["forecast_horizon"]
              // cfg["realized_vol"]["subsampling_freq"])
    purge_bars = h_bars

    all_metrics, regime_rows, dm_rows = [], [], []

    for symbol in cfg["data"]["symbols"]:
        safe = symbol.replace("/", "-")
        tda_path = proc_dir / f"{safe}_tda_features.csv.gz"
        rv_path  = proc_dir / f"{safe}_rv_features.csv.gz"
        has_tda  = tda_path.exists()
        path = tda_path if has_tda else rv_path
        if not path.exists():
            print(f"Missing {path}, skipping.")
            continue

        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        tda_cols  = [c for c in TDA_FEATURES if c in df.columns]
        ctrl_cols = [c for c in CONTROL_FEATURES if c in df.columns]
        print(f"\n=== {symbol}: {len(df):,} rows "
              f"({'TDA-augmented' if tda_cols else 'baseline only'}"
              f"{', +controls' if ctrl_cols else ''}) ===")

        models = {
            "HAR-RV":  lambda: HARRV(HAR_FEATURES),
            "XGBoost": lambda: XGBBaseline(HAR_FEATURES),
        }
        if tda_cols:
            har_tda = HAR_FEATURES + [c for c in HAR_TDA_EXTRA if c in tda_cols]
            xgb_tda = HAR_FEATURES + tda_cols
            models["HAR+TDA"]     = lambda f=har_tda: HARRV(f)
            models["XGBoost+TDA"] = lambda f=xgb_tda: XGBBaseline(f)
            if ctrl_cols:
                har_c  = HAR_FEATURES + ctrl_cols
                xgb_c  = HAR_FEATURES + ctrl_cols
                har_ct = HAR_FEATURES + ctrl_cols + [c for c in HAR_TDA_EXTRA if c in tda_cols]
                xgb_ct = HAR_FEATURES + ctrl_cols + tda_cols
                models["HAR+RV1"]         = lambda f=har_c:  HARRV(f)
                models["XGBoost+RV1"]     = lambda f=xgb_c:  XGBBaseline(f)
                models["HAR+RV1+TDA"]     = lambda f=har_ct: HARRV(f)
                models["XGBoost+RV1+TDA"] = lambda f=xgb_ct: XGBBaseline(f)

        results = {}
        for name, factory in models.items():
            result = walk_forward(factory, name, symbol, df,
                                  oos_start=oos_start, step_days=step_days,
                                  purge_bars=purge_bars)
            results[name] = result

            pred_df = pd.DataFrame({
                "actual_rv":    result.actuals,
                "predicted_rv": result.predictions,
            })
            tag = name.lower().replace("-", "_").replace("+", "_")
            pred_df.to_csv(out_dir / f"{safe}_{tag}_preds.csv.gz", compression="gzip")

            m = result.overall_metrics()
            all_metrics.append(m)
            regime_rows.extend(result.regime_metrics(turbulent))
            print(f"  {name:12s}: QLIKE={m['qlike']:.6f}  MSE-log={m['mse_log']:.6f}  n={m['n_obs']:,}")

        # Diebold–Mariano comparisons (HAC lag = horizon-1). The last two ISOLATE
        # the topology effect by holding the 1-min RV control fixed on both sides.
        dm_pairs = [
            ("HAR-RV",      "HAR+TDA"),          # naive linear gain (frequency-confounded)
            ("XGBoost",     "XGBoost+TDA"),      # naive nonlinear gain (frequency-confounded)
            ("HAR-RV",      "HAR+RV1"),          # pure frequency gain (no topology)
            ("XGBoost",     "XGBoost+RV1"),      # pure frequency gain (no topology)
            ("HAR+RV1",     "HAR+RV1+TDA"),      # TOPOLOGY beyond control (linear)
            ("XGBoost+RV1", "XGBoost+RV1+TDA"),  # TOPOLOGY beyond control (nonlinear) ← key
        ]
        for base, tda in dm_pairs:
            if base not in results or tda not in results:
                continue
            rb, rt = results[base], results[tda]
            idx = rb.actuals.index.intersection(rt.actuals.index)
            act = rb.actuals.reindex(idx).values
            lb = qlike_pointwise(act, rb.predictions.reindex(idx).values)
            lt = qlike_pointwise(act, rt.predictions.reindex(idx).values)
            dm = diebold_mariano(lb, lt, h=h_bars)
            dm_rows.append({
                "symbol":     symbol,
                "comparison": f"{base} → {tda}",
                "qlike_base": float(np.mean(lb)),
                "qlike_tda":  float(np.mean(lt)),
                **dm,
            })
            arrow = "↓ better" if dm["mean_diff"] > 0 else "↑ worse"
            print(f"  DM [{base} → {tda}]: Δ={dm['mean_diff']:+.5f} ({arrow})  "
                  f"stat={dm['dm_stat']:+.2f}  p={dm['p_value']:.4f}  → {dm['verdict']}")

    summary = pd.DataFrame(all_metrics)
    summary.to_csv(out_dir / "model_summary.csv", index=False)

    if regime_rows:
        pd.DataFrame(regime_rows).to_csv(out_dir / "regime_breakdown.csv", index=False)
        print(f"\nRegime breakdown saved → {out_dir}/regime_breakdown.csv")

    dm_df = pd.DataFrame(dm_rows)
    if not dm_df.empty:
        dm_df.to_csv(out_dir / "dm_tests.csv", index=False)
        print(f"Diebold–Mariano tests saved → {out_dir}/dm_tests.csv")

    print(f"Model summary saved → {out_dir}/model_summary.csv")
    return summary, dm_df


# Back-compat alias.
def run_baselines(cfg_path: str = "config.yaml", results_dir: str = "results") -> pd.DataFrame:
    return run_evaluation(cfg_path, results_dir)[0]


if __name__ == "__main__":
    summary, dm = run_evaluation()
    print("\n=== Model Results ===")
    print(summary.to_string(index=False))
    if not dm.empty:
        print("\n=== Diebold–Mariano (TDA vs baseline, QLIKE) ===")
        cols = ["symbol", "comparison", "qlike_base", "qlike_tda",
                "mean_diff", "dm_stat", "p_value", "verdict"]
        print(dm[cols].to_string(index=False))
