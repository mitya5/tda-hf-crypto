"""
significance.py
---------------
Diebold–Mariano (1995) test of equal predictive accuracy, with the
Harvey–Leybourne–Newbold (1997) small-sample correction.

WHY THIS MATTERS HERE:
  HAR-RV is famously hard to beat, so any QLIKE improvement from TDA features
  will be small. A raw drop like 0.639 → 0.631 is meaningless without a test of
  whether the loss differential is statistically distinguishable from zero. The
  DM test does exactly that, accounting for the heavy serial correlation of
  intraday volatility losses.

OVERLAPPING FORECASTS:
  Our target is RV over the next 30 minutes, emitted every 5 minutes, so adjacent
  forecasts overlap by h = 30/5 = 6 bars. Overlapping multi-step forecasts induce
  MA(h-1) autocorrelation in the loss differential, so we estimate the long-run
  variance with a Newey–West (Bartlett) HAC sum out to lag h-1.

CONVENTION:
  d_t = loss_baseline_t - loss_tda_t.  A POSITIVE mean differential ⇒ the TDA
  model has lower loss ⇒ TDA helps. p-value is two-sided.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def qlike_pointwise(rv_true: np.ndarray, rv_pred: np.ndarray) -> np.ndarray:
    """Per-observation QLIKE loss (so it can be fed to the DM test)."""
    rv_true = np.clip(np.asarray(rv_true, float), 1e-20, None)
    rv_pred = np.clip(np.asarray(rv_pred, float), 1e-20, None)
    return rv_true / rv_pred - np.log(rv_true / rv_pred) - 1.0


def se_pointwise(rv_true: np.ndarray, rv_pred: np.ndarray) -> np.ndarray:
    """Per-observation squared error on log-RV."""
    rv_true = np.clip(np.asarray(rv_true, float), 1e-20, None)
    rv_pred = np.clip(np.asarray(rv_pred, float), 1e-20, None)
    return (np.log(rv_true) - np.log(rv_pred)) ** 2


def diebold_mariano(loss_base: np.ndarray, loss_tda: np.ndarray, h: int = 1) -> dict:
    """
    DM test with HLN small-sample correction.

    Parameters
    ----------
    loss_base, loss_tda : per-observation losses for the two models (same index).
    h : forecast horizon in bars; HAC lag truncation is h-1.

    Returns dict with the statistic, two-sided p-value, mean differential and a
    human-readable verdict. Positive mean_diff ⇒ TDA model is better.
    """
    d = np.asarray(loss_base, float) - np.asarray(loss_tda, float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 10:
        return {"dm_stat": np.nan, "p_value": np.nan, "mean_diff": np.nan,
                "n": n, "verdict": "insufficient data"}

    dbar = d.mean()
    dc = d - dbar

    # Newey–West long-run variance: gamma_0 + 2 Σ_{k=1}^{h-1} gamma_k
    gamma0 = float(dc @ dc) / n
    lrv = gamma0
    for k in range(1, max(h, 1)):
        if k >= n:
            break
        gamma_k = float(dc[k:] @ dc[:-k]) / n
        lrv += 2.0 * gamma_k
    lrv = max(lrv, 1e-30)

    dm = dbar / np.sqrt(lrv / n)

    # Harvey–Leybourne–Newbold correction + Student-t reference
    hln = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    dm_hln = dm * hln
    p = 2.0 * (1.0 - stats.t.cdf(abs(dm_hln), df=n - 1))

    if p < 0.05:
        verdict = "TDA significantly BETTER" if dbar > 0 else "TDA significantly WORSE"
    else:
        verdict = "no significant difference"

    return {
        "dm_stat":   float(dm_hln),
        "p_value":   float(p),
        "mean_diff": float(dbar),
        "n":         int(n),
        "verdict":   verdict,
    }
