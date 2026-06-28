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

# Trailing windows (minutes) for 1-min realized variance controls.
CONTROL_WINDOWS = [15, 60]
CONTROL_FEATURES = [f"log_rv1_{w}" for w in CONTROL_WINDOWS]


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
