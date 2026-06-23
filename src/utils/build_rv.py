"""
build_rv.py
-----------
Build the realized-volatility target and HAR-RV lag features from clean 1-min
OHLCV parquet files.

Key design choices (document and justify in the paper):
  - We sub-sample to 5-minute bars before computing RV to sit outside the
    microstructure-noise regime revealed by the signature plot.
  - We use the Yang–Zhang estimator on the 5-min bars: it exploits OHLC
    information and is robust to both overnight gaps and bid–ask bounce,
    making it more efficient than simple close-to-close RV.
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


# ── Estimators ───────────────────────────────────────────────────────────────

def yang_zhang_rv(df: pd.DataFrame) -> pd.Series:
    """
    Yang–Zhang (2000) realized volatility estimator.

    Combines overnight return, open-to-close, and Rogers–Satchell components.
    Returns annualised volatility per bar.

    Parameters
    ----------
    df : DataFrame with columns open, high, low, close at a fixed frequency.
    """
    log_ho = np.log(df["high"] / df["open"])
    log_lo = np.log(df["low"]  / df["open"])
    log_co = np.log(df["close"] / df["open"])

    # Rogers–Satchell component (intraday)
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    # Open-to-close
    log_oc = np.log(df["close"] / df["open"])
    mean_oc = log_oc.mean()
    oc_var = ((log_oc - mean_oc) ** 2).mean()

    # k constant (annualisation correction; σ = 0.34 is YZ default)
    k = 0.34 / (1.34 + (len(df) + 1) / (len(df) - 1))

    var_yz = k * oc_var + (1 - k) * rs.mean()
    return max(var_yz, 0.0)  # numerical floor


def rolling_rv_5min(df_5m: pd.DataFrame, window_bars: int) -> pd.Series:
    """
    Compute rolling Yang–Zhang RV over a backward-looking window of bars.
    Returns a Series aligned with df_5m.index (each value = RV of prior `window` bars).
    """
    rv_vals = []
    for i in range(len(df_5m)):
        start = max(0, i - window_bars + 1)
        slice_ = df_5m.iloc[start : i + 1]
        if len(slice_) < 2:
            rv_vals.append(np.nan)
        else:
            rv_vals.append(yang_zhang_rv(slice_))
    return pd.Series(rv_vals, index=df_5m.index)


def forward_rv(df_5m: pd.DataFrame, horizon_bars: int) -> pd.Series:
    """
    Forward-looking RV target: RV over the NEXT `horizon_bars` bars.
    Uses .shift(-horizon_bars) so no look-ahead at inference time.
    """
    rv_vals = []
    for i in range(len(df_5m)):
        end = min(i + horizon_bars, len(df_5m))
        slice_ = df_5m.iloc[i:end]
        if len(slice_) < 2:
            rv_vals.append(np.nan)
        else:
            rv_vals.append(yang_zhang_rv(slice_))
    return pd.Series(rv_vals, index=df_5m.index)


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
    freq = cfg["realized_vol"]["subsampling_freq"]
    df_5m = df_1m.resample(f"{freq}min", label="right", closed="right").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["close"])
    print(f"[{symbol}] 5-min bars: {len(df_5m):,}")

    horizon_min = cfg["realized_vol"]["forecast_horizon"]
    horizon_bars = horizon_min // freq   # e.g. 30 min / 5 min = 6 bars

    # HAR lag windows (in minutes → convert to bars)
    har = cfg["har"]
    lag_windows_min = {
        "5min":   5,
        "30min":  30,
        "60min":  60,
        "240min": 240,
        "480min": 480,
    }

    print(f"[{symbol}] Computing forward RV target (horizon={horizon_min}min)…")
    rv_target = forward_rv(df_5m, horizon_bars)

    print(f"[{symbol}] Computing HAR lag features…")
    features = {"rv_target": rv_target}
    for name, win_min in lag_windows_min.items():
        win_bars = max(1, win_min // freq)
        rv_lag = rolling_rv_5min(df_5m, win_bars)
        features[f"rv_lag_{name}"]     = rv_lag
        features[f"log_rv_lag_{name}"] = np.log(rv_lag.clip(lower=1e-12))

    out = pd.DataFrame(features, index=df_5m.index)
    out = out.dropna()

    # Basic sanity check
    assert (out["rv_target"] >= 0).all(), "Negative RV values found — check estimator."
    print(f"[{symbol}] Final rows after dropping NaNs: {len(out):,}")
    print(f"[{symbol}] Date range: {out.index.min().date()} → {out.index.max().date()}")
    print(f"[{symbol}] RV target stats:\n{out['rv_target'].describe()}\n")

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
