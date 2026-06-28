"""
build_tda.py
------------
Build TDA-augmented feature matrices: takes the existing RV feature grid
(data/processed/<SYM>_rv_features.csv.gz) and joins on the causal persistent-
homology features computed from the 1-minute returns, writing
data/processed/<SYM>_tda_features.csv.gz.

The output is a strict superset of the RV feature file (same rows, same index),
so baseline and TDA models are evaluated on identical observations — a clean
head-to-head and a valid Diebold–Mariano comparison.

HIGH FREQUENCY: features are computed from 1-minute returns (never downsampled).
The 5-min RV grid is used only to decide *where* to emit a feature row.

Usage
-----
  python -m src.features.build_tda                 # all symbols, full span
  python -m src.features.build_tda --limit 3000    # quick validation slice
  python -m src.features.build_tda --start 2022-11-01 --end 2022-11-30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.tda import TDAConfig, compute_tda_features, TDA_FEATURES
from src.features.controls import add_frequency_controls, CONTROL_FEATURES


def build_symbol(symbol: str, cfg: dict, tcfg: TDAConfig,
                 start: str | None, end: str | None, limit: int | None,
                 progress: bool = True, n_jobs: int = 1) -> pd.DataFrame:
    raw_dir  = Path(cfg["data"]["raw_dir"])
    proc_dir = Path(cfg["data"]["proc_dir"])
    safe = symbol.replace("/", "-")

    rv_path = proc_dir / f"{safe}_rv_features.csv.gz"
    parquet = raw_dir / f"{safe}_1m.parquet"
    if not rv_path.exists():
        raise FileNotFoundError(f"Missing {rv_path}; run src/utils/build_rv.py first.")
    if not parquet.exists():
        raise FileNotFoundError(f"Missing {parquet}; run src/utils/fetch_data.py first.")

    rv = pd.read_csv(rv_path, index_col=0, parse_dates=True)
    if rv.index.tz is None:
        rv.index = rv.index.tz_localize("UTC")

    grid = rv.index
    if start is not None:
        grid = grid[grid >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        grid = grid[grid <= pd.Timestamp(end, tz="UTC")]
    if limit is not None:
        grid = grid[:limit]

    print(f"[{symbol}] 1-min source: {parquet}")
    close = pd.read_parquet(parquet)["close"]
    if close.index.tz is None:
        close.index = close.index.tz_localize("UTC")
    close = close.sort_index()

    print(f"[{symbol}] computing TDA features for {len(grid):,} bars "
          f"(window={tcfg.window_size}m, m={tcfg.embedding_dim}, tau={tcfg.embedding_delay}, "
          f"jobs={n_jobs})…")
    tda = compute_tda_features(close, grid, tcfg, progress=progress, n_jobs=n_jobs)

    deg = tda["tda_degenerate"].mean()
    print(f"[{symbol}] degenerate (flat/illiquid) windows: {deg:.1%}")

    # Non-topological 1-min realized-vol controls (to isolate the topology effect).
    controls = add_frequency_controls(close, grid, ret_clip=tcfg.ret_clip)

    out = rv.loc[grid].join(tda, how="left").join(controls, how="left")
    # any unexpected gaps in the join → zero-structure, flagged degenerate
    miss = out["tda_degenerate"].isna()
    if miss.any():
        out.loc[miss, TDA_FEATURES] = 0.0
        out.loc[miss, "tda_degenerate"] = 1.0

    out_path = proc_dir / f"{safe}_tda_features.csv.gz"
    out.to_csv(out_path, compression="gzip")
    print(f"[{symbol}] saved → {out_path}  ({len(out):,} rows, "
          f"{len(TDA_FEATURES)} TDA cols)\n")
    return out


def main():
    ap = argparse.ArgumentParser(description="Build TDA-augmented feature matrices")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--symbols", nargs="*", default=None, help="override config symbols")
    ap.add_argument("--start", default=None, help="UTC date, e.g. 2022-11-01")
    ap.add_argument("--end",   default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap #bars (quick test)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel workers (try 8)")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tcfg = TDAConfig.from_yaml(cfg)

    symbols = args.symbols or cfg["data"]["symbols"]
    for symbol in symbols:
        try:
            build_symbol(symbol, cfg, tcfg, args.start, args.end, args.limit,
                         progress=not args.no_progress, n_jobs=args.jobs)
        except FileNotFoundError as e:
            print(f"Skipping {symbol}: {e}")

    print("Done. TDA feature files written to", cfg["data"]["proc_dir"])


if __name__ == "__main__":
    main()
