"""
fetch_data.py
-------------
Download 1-minute OHLCV bars from Binance for all symbols in config.yaml
and save each symbol as a Parquet file in data/raw/.

Usage:
    python src/utils/fetch_data.py              # uses config.yaml defaults
    python src/utils/fetch_data.py --symbol BTC/USDT --start 2022-01-01

Notes
-----
- Binance caps requests at 1000 bars per call; this script pages through them.
- No API key required for public OHLCV data.
- Crypto is 24/7 with no calendar gaps, but ticks can still be missing at the
  1-minute level (exchange maintenance, etc.); gaps are logged and forward-filled
  in the cleaning step.
- A volatility signature plot is written to data/raw/<symbol>_sig_plot.png so you
  can sanity-check microstructure noise before building RV targets.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# Allow `python src/utils/fetch_data.py` to find the sibling cleaning module.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.cleaning import clean_ohlc_spikes


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def fetch_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    limit: int = 1000,
) -> pd.DataFrame:
    """Page through Binance OHLCV endpoint and return a clean DataFrame."""
    all_bars = []
    cursor = since_ms

    pbar = tqdm(desc=f"{symbol} {timeframe}", unit=" bars", dynamic_ncols=True)

    while cursor < until_ms:
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        except ccxt.NetworkError as e:
            print(f"\n[retry] Network error: {e}. Sleeping 10s…")
            time.sleep(10)
            continue
        except ccxt.ExchangeError as e:
            print(f"\n[abort] Exchange error: {e}")
            break

        if not bars:
            break

        all_bars.extend(bars)
        pbar.update(len(bars))

        last_ts = bars[-1][0]
        if last_ts == cursor:          # no progress → done
            break
        cursor = last_ts + 1           # next page starts 1 ms after last bar

        # Binance public rate limit: ~1200 req/min; be conservative
        time.sleep(0.05)

    pbar.close()

    if not all_bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(
        all_bars, columns=["timestamp_ms", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.drop(columns=["timestamp_ms"]).set_index("timestamp")
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def clean_ohlcv(df: pd.DataFrame, symbol: str, timeframe: str = "1min") -> pd.DataFrame:
    """
    Fill gaps and remove outliers.

    - Reindex to a complete minute grid so downstream code can rely on uniform spacing.
    - Forward-fill short gaps (≤ 5 bars) — consistent with market microstructure practice.
    - Log gaps > 5 bars as warnings (likely exchange maintenance).
    - Remove obvious price spikes: any bar where |log return| > 0.10 (10%) is
      flagged; close is replaced with VWAP-style midpoint of neighboring bars.
    """
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=timeframe, tz="UTC")
    df = df.reindex(full_idx)

    # Log gaps before filling
    gap_mask = df["close"].isna()
    if gap_mask.any():
        gap_runs = (gap_mask != gap_mask.shift()).cumsum()[gap_mask]
        run_lengths = gap_runs.value_counts().sort_index()
        long_gaps = run_lengths[run_lengths > 5]
        if not long_gaps.empty:
            print(f"\n[{symbol}] Warning: {len(long_gaps)} gap(s) > 5 bars found:")
            for run_id in long_gaps.index:
                gap_start = gap_runs[gap_runs == run_id].index[0]
                print(f"    starting {gap_start}, length {run_lengths[run_id]} bars")

    df = df.ffill(limit=5)

    # Spike detection on close-to-close log returns (catches bad close prints).
    log_ret = np.log(df["close"]).diff()
    spike_mask = log_ret.abs() > 0.10
    if spike_mask.any():
        n_spikes = spike_mask.sum()
        print(f"[{symbol}] Flagging {n_spikes} price spike(s) (|log ret| > 10%) — interpolating.")
        df.loc[spike_mask, ["open", "high", "low", "close"]] = np.nan
        df[["open", "high", "low", "close"]] = (
            df[["open", "high", "low", "close"]].interpolate(method="time")
        )

    # Any remaining NaNs (e.g. at very start) → drop
    df = df.dropna(subset=["close"])

    # Repair bad high/low ticks — the close filter above cannot see these, but
    # the Yang–Zhang RV estimator reads high/low directly, so they must go.
    df = clean_ohlc_spikes(df)
    n_clipped = df.attrs.get("n_wicks_clipped", 0)
    if n_clipped:
        print(f"[{symbol}] Clipped {n_clipped} bad high/low wick(s).")

    return df


def vol_signature_plot(df: pd.DataFrame, symbol: str, out_path: Path) -> None:
    """
    Volatility signature plot: RV vs. sampling frequency.
    Flat → that frequency is in the 'safe zone'; rising as freq increases → noise regime.
    """
    freqs_min = [1, 2, 3, 5, 10, 15, 30, 60]
    rvs = []
    for f in freqs_min:
        sampled = df["close"].resample(f"{f}min").last().dropna()
        log_ret = np.log(sampled).diff().dropna()
        rvs.append(np.sqrt((log_ret**2).mean()) * np.sqrt(252 * 24 * 60 / f))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(freqs_min, rvs, "o-", color="steelblue")
    ax.axvline(5, color="red", linestyle="--", label="5-min (our choice)")
    ax.set_xlabel("Sampling interval (minutes)")
    ax.set_ylabel("Annualised RV")
    ax.set_title(f"{symbol} — Volatility Signature Plot")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[{symbol}] Signature plot saved → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(cfg_path: str = "config.yaml", symbol_filter: str | None = None,
         start_override: str | None = None, end_override: str | None = None) -> None:

    cfg = load_config(cfg_path)
    raw_dir = Path(cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    exchange_id = cfg["data"]["exchange"]   # e.g. "binanceus"
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})

    symbols = cfg["data"]["symbols"]
    if symbol_filter:
        symbols = [s for s in symbols if s == symbol_filter]

    start = start_override or cfg["data"]["start_date"]
    end   = end_override   or cfg["data"]["end_date"]

    since_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    until_ms = int(pd.Timestamp(end,   tz="UTC").timestamp() * 1000)

    for symbol in symbols:
        safe_name = symbol.replace("/", "-")
        out_parquet = raw_dir / f"{safe_name}_1m.parquet"

        if out_parquet.exists():
            existing = pd.read_parquet(out_parquet)
            print(f"[{symbol}] Already have {len(existing):,} bars "
                  f"({existing.index.min().date()} → {existing.index.max().date()}).")
            # Extend if needed
            last_ts_ms = int(existing.index.max().timestamp() * 1000) + 1
            if last_ts_ms >= until_ms:
                print(f"[{symbol}] Up to date, skipping.")
                continue
            print(f"[{symbol}] Extending from {existing.index.max().date()}…")
            new_df = fetch_ohlcv(exchange, symbol, cfg["data"]["timeframe"],
                                 last_ts_ms, until_ms)
            df = pd.concat([existing, new_df])
        else:
            print(f"\n[{symbol}] Fetching {start} → {end}…")
            df = fetch_ohlcv(exchange, symbol, cfg["data"]["timeframe"],
                             since_ms, until_ms)

        print(f"[{symbol}] Raw: {len(df):,} bars. Cleaning…")
        df = clean_ohlcv(df, symbol)
        print(f"[{symbol}] Clean: {len(df):,} bars "
              f"({df.index.min().date()} → {df.index.max().date()}).")

        df.to_parquet(out_parquet, compression="zstd")
        print(f"[{symbol}] Saved → {out_parquet}")

        # Signature plot on a random 90-day sub-sample to keep it fast
        sample_start = pd.Timestamp("2022-01-01", tz="UTC")
        sample_end   = pd.Timestamp("2022-04-01", tz="UTC")
        sample = df.loc[sample_start:sample_end]
        if len(sample) > 1000:
            vol_signature_plot(
                sample, symbol,
                raw_dir / f"{safe_name}_sig_plot.png"
            )

    print("\nDone. All symbols fetched and cleaned.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Binance OHLCV data")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--symbol", default=None, help="e.g. 'BTC/USDT'")
    parser.add_argument("--start",  default=None, help="e.g. '2022-01-01'")
    parser.add_argument("--end",    default=None, help="e.g. '2023-01-01'")
    args = parser.parse_args()
    main(args.config, args.symbol, args.start, args.end)
