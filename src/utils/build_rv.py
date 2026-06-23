"""
build_rv.py
-----------
Build the realized-volatility target and HAR-RV lag features from clean 1-min
OHLCV parquet files.

Key design choices (document and justify in the paper):
  - We sub-sample to 5-minute bars before computing RV to sit outside the
    microstructure-noise regime revealed by the signature plot.
  - We use a vectorized Rogers–Satchell / Yang–Zhang estimator on the 5-min
    bars: exploits OHLC information and is more efficient than close-to-close RV.
  - Target: RV over the NEXT `horizon_min` minutes (30 by default), i.e.
    the variable we want to forecast — strictly forward-looking, no look-ahead.

Output: data/processed/<SYMBOL>_rv_features.parquet
  Each row is a 5-minute bar with columns:
    rv_target          : forward-looking RV over next `horizon` bars
    rv_lag_{n}min      : HAR-style backward-looking RV over past n minutes
    log_rv_lag_{n}min  : log-transformed lags (often more Gaussian in practice)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ── Vectorized estimator ─────────────────────────────────────────────────────

def _yz_components(df_5m: pd.DataFrame):
    """
    Pre-compute the per-bar scalar components needed for the Yang–Zhang estimator.
    All are Series aligned with df_5m.index.
    """
    log_ho = np.log(df_5m["high"]  / df_5m["open"])
    log_lo = np.log(df_5m["low"]   / df_5m["open"])
    log_co = np.log(df_5m["close"] / df_5m["open"])

    # Rogers–Satchell term (intraday variance contribution per bar)
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    rs = rs.clip(lower=0)   # numerical floor

    # Open-to-close log return (same as log_co here since we use open as base)
    log_oc = log_co

    return rs, log_oc


def rolling_yz_rv(df_5m: pd.DataFrame, window: int) -> pd.Series:
    """
    Vectorized rolling Yang–Zhang RV over `window` bars.

    For a fixed window n, k is constant:
        k = 0.34 / (1.34 + (n+1)/(n-1))
    So:
        var_yz = k * rolling_var(log_oc) + (1-k) * rolling_mean(rs)
    """
    if window < 2:
        raise ValueError("window must be >= 2")

    rs, log_oc = _yz_components(df_5m)

    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    # rolling variance of open-to-close (ddof=1 matches sample variance)
    oc_var = log_oc.rolling(window, min_periods=window).var(ddof=1)
    rs_mean = rs.rolling(window, min_periods=window).mean()

    rv = (k * oc_var + (1 - k) * rs_mean).clip(lower=0)
    return rv


def forward_yz_rv(df_5m: pd.DataFrame, horizon: int) -> pd.Series:
    """
    Forward-looking RV target: RV computed over the NEXT `horizon` bars.

    We compute a backward-looking rolling RV of length `horizon` and then
    shift it backward by `horizon` bars — this is equivalent to a strictly
    causal forward window and introduces zero look-ahead.
    """
    backward = rolling_yz_rv(df_5m, horizon)
    # shift(-horizon): at time t, the value is the RV of bars [t, t+horizon)
    return backward.shift(-horizon)


# ── Main ─────────────────────────────────────────────────────────────────────

def build_features(symbol: str, cfg: dict) -> pd.DataFrame:
    raw_dir  = Path(cfg["data"]["raw_dir"])
    proc_dir = Path(cfg["data"]["proc_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)

    safe_name = symbol.replace("/", "-")
    parquet_in = raw_dir / f"{safe_name}_1m.parquet"
    if not parquet_in.exists():
        raise FileNotFoundError(
            f"Missing {parquet_in}. Run src/utils/fetch_data.py first."
        )

    print(f"[{symbol}] Loading 1-min data…")
    df_1m = pd.read_parquet(parquet_in)

    # Sub-sample to 5-minute bars
    freq = cfg["realized_vol"]["subsampling_freq"]   # 5
    df_5m = df_1m.resample(f"{freq}min", label="right", closed="right").agg(
        open=("open",   "first"),
        high=("high",   "max"),
        low=("low",     "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["close"])
    print(f"[{symbol}] 5-min bars: {len(df_5m):,}")

    horizon_min  = cfg["realized_vol"]["forecast_horizon"]   # 30
    horizon_bars = horizon_min // freq                       # 6

    # HAR lag windows (minutes → bars)
    lag_windows = {
        "5min":   5   // freq,   # 1  bar
        "30min":  30  // freq,   # 6  bars
        "60min":  60  // freq,   # 12 bars
        "240min": 240 // freq,   # 48 bars
        "480min": 480 // freq,   # 96 bars
    }
    # Enforce minimum window of 2
    lag_windows = {k: max(v, 2) for k, v in lag_windows.items()}

    print(f"[{symbol}] Computing forward RV target (horizon={horizon_min} min = {horizon_bars} bars)…")
    rv_target = forward_yz_rv(df_5m, horizon_bars)

    print(f"[{symbol}] Computing HAR lag features…")
    features = {"rv_target": rv_target}
    for name, win_bars in lag_windows.items():
        rv_lag = rolling_yz_rv(df_5m, win_bars)
        features[f"rv_lag_{name}"]     = rv_lag
        features[f"log_rv_lag_{name}"] = np.log(rv_lag.clip(lower=1e-12))

    out = pd.DataFrame(features, index=df_5m.index)
    out = out.dropna()

    assert (out["rv_target"] >= 0).all(), "Negative RV values — check estimator."
    print(f"[{symbol}] Final rows: {len(out):,}")
    print(f"[{symbol}] Date range: {out.index.min().date()} → {out.index.max().date()}")
    print(f"[{symbol}] rv_target stats:\n{out['rv_target'].describe()}\n")

    out_path = proc_dir / f"{safe_name}_rv_features.parquet"
    out.to_parquet(out_path, compression="zstd")
    print(f"[{symbol}] Saved → {out_path}")
    return out


def main(cfg_path: str = "config.yaml") -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    for symbol in cfg["data"]["symbols"]:
        try:
            build_features(symbol, cfg)
        except FileNotFoundError as e:
            print(f"Skipping {symbol}: {e}")

    print("\nDone. RV feature files written to", cfg["data"]["proc_dir"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RV features from clean OHLCV")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
