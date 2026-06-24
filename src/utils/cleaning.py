"""
cleaning.py
-----------
Shared OHLC data-cleaning helpers used by both the fetch step (fetch_data.py)
and the feature-build step (build_rv.py).

The key routine is ``clean_ohlc_spikes``, which removes *bad high/low ticks* —
single erroneous prints in the high or low column that are common on thin
exchanges. These are invisible to a close-to-close return filter (the close is
fine), but the Yang–Zhang RV estimator reads the high and low directly, so a
single bad wick (e.g. a BTC high of $138,070 while the bar trades at $28,800)
produces a realized-variance value 100–1000× too large and dominates every loss
metric. See build_rv.py for how this feeds the RV target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_ohlc_spikes(df: pd.DataFrame, wick_threshold: float = 0.15) -> pd.DataFrame:
    """
    Repair implausible high/low wicks and enforce OHLC structural validity.

    A bar's high may not exceed max(open, close) by more than ``wick_threshold``
    (and symmetrically for the low). Crypto 1-minute bars essentially never have
    a *genuine* intrabar wick beyond ~15% for BTC/ETH, so anything larger is a
    bad print. Offending wicks are clipped back to the candle body — this removes
    the spurious range while preserving the close-to-close move (real volatility
    still registers through the open/close path of surrounding bars).

    Parameters
    ----------
    df : DataFrame with columns open, high, low, close (any extras pass through).
    wick_threshold : max allowed wick beyond the body, as a fraction (0.15 = 15%).

    Returns
    -------
    A copy of ``df`` with cleaned high/low and the number of repaired wicks
    reported on the returned object as ``df.attrs['n_wicks_clipped']``.
    """
    df = df.copy()
    body_hi = df[["open", "close"]].max(axis=1)
    body_lo = df[["open", "close"]].min(axis=1)

    bad_hi = df["high"] > body_hi * (1 + wick_threshold)
    bad_lo = df["low"] < body_lo * (1 - wick_threshold)

    df.loc[bad_hi, "high"] = body_hi[bad_hi]
    df.loc[bad_lo, "low"] = body_lo[bad_lo]

    # Structural safety net: high must be the max and low the min of the bar.
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)

    df.attrs["n_wicks_clipped"] = int(bad_hi.sum() + bad_lo.sum())
    return df
