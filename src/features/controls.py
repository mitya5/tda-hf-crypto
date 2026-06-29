"""
controls.py
-----------
Non-topological "frequency controls": realized variance computed from the SAME
1-minute returns the TDA pipeline embeds, over trailing windows, evaluated
causally on the 5-min RV grid.

WHY THIS EXISTS:
  The TDA features clearly improve the forecasts, but several of them (e.g.
  tda_total_pers_h0) are ~0.8 correlated with 1-minute realized volatility. The
  baselines only see 5-minute Yang–Zhang RV lags, so part of the TDA "win" is
  simply that topology smuggles in 1-minute information the baseline lacks — a
  FREQUENCY effect, not a TOPOLOGY effect. Adding these controls to the baseline
  lets the Diebold–Mariano test isolate the genuinely topological contribution:
      (base + 1-min RV)  vs  (base + 1-min RV + TDA).
  If topology still helps on top of the control, the claim is defensible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.cleaning import clean_ohlc_spikes

# Trailing windows (minutes) for 1-min realized variance controls.
CONTROL_WINDOWS = [15, 60]
CONTROL_FEATURES = [f"log_rv1_{w}" for w in CONTROL_WINDOWS]

# Jump/outlier-ROBUST 1-min volatility controls, for the stronger robustness test:
#   bipower variation (jump-robust) and realized range (high/low based). These
#   target the exact property that total-persistence-H0 might exploit — robustness
#   to single freak prints — so beating them is a stronger claim for topology.
ROBUST_CONTROL_FEATURES = (
    [f"log_bv1_{w}" for w in CONTROL_WINDOWS] +   # bipower variation
    [f"log_rr1_{w}" for w in CONTROL_WINDOWS]     # realized range
)


def add_frequency_controls(close_1m: pd.Series, grid_index: pd.DatetimeIndex,
                           ret_clip: float = 0.20) -> pd.DataFrame:
    """
    Causal 1-minute realized-variance controls aligned to `grid_index`.

    log_rv1_W = log( Σ r_i²  over the W 1-min returns in (t-W, t] ),
    clipped/floored exactly like the RV pipeline so it is comparable in scale.
    Strictly backward-looking — same (t-W, t] convention as the TDA windows.
    """
    r = np.log(close_1m).diff().clip(-ret_clip, ret_clip)
    r2 = (r ** 2)
    out = {}
    for w in CONTROL_WINDOWS:
        rv = r2.rolling(w, min_periods=max(w // 2, 2)).sum()
        # grid timestamps fall on 1-min boundaries → exact reindex (no look-ahead)
        rv_grid = rv.reindex(grid_index)
        out[f"log_rv1_{w}"] = np.log(rv_grid.clip(lower=1e-12))
    df = pd.DataFrame(out, index=grid_index)
    return df


def add_robust_controls(ohlc_1m: pd.DataFrame, grid_index: pd.DatetimeIndex,
                        wick_threshold: float = 0.15, ret_clip: float = 0.20) -> pd.DataFrame:
    """
    Jump/outlier-robust 1-minute volatility controls, aligned causally to
    `grid_index`. Both are strictly backward-looking over (t-W, t]:

      Bipower variation (Barndorff-Nielsen & Shephard) — jump-robust:
          BV_W(t) = (pi/2) * Σ |r_i| |r_{i-1}|    over the W returns ending at t.
      Realized range (Parkinson / Christensen-Podolskij) — uses intrabar high/low,
      more efficient and noise-robust, on REPAIRED wicks (clean_ohlc_spikes):
          RR_W(t) = Σ (ln H_i - ln L_i)^2 / (4 ln 2)   over the W bars ending at t.
    """
    ohlc = clean_ohlc_spikes(ohlc_1m, wick_threshold=wick_threshold)

    r = np.log(ohlc["close"]).diff().clip(-ret_clip, ret_clip)
    bp = (r.abs() * r.abs().shift(1)) * (np.pi / 2.0)          # bipower terms
    rng = (np.log(ohlc["high"]) - np.log(ohlc["low"])) ** 2 / (4.0 * np.log(2.0))

    out = {}
    for w in CONTROL_WINDOWS:
        bv = bp.rolling(w, min_periods=max(w // 2, 2)).sum()
        rr = rng.rolling(w, min_periods=max(w // 2, 2)).sum()
        out[f"log_bv1_{w}"] = np.log(bv.reindex(grid_index).clip(lower=1e-12))
        out[f"log_rr1_{w}"] = np.log(rr.reindex(grid_index).clip(lower=1e-12))
    return pd.DataFrame(out, index=grid_index)
