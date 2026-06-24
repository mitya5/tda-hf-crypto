# TDA-Augmented High-Frequency Volatility Forecasting in Crypto Markets

**REU Paper Project — University of Chicago, Summer 2026**

## Research Question

Do persistent-homology (TDA) features of high-frequency crypto returns improve short-horizon realized-volatility forecasts beyond HAR-RV and gradient-boosted tree baselines?

This project extends [Souto (2023)](https://doi.org/10.1016/j.jfds.2023.100107) — which demonstrated TDA-augmented volatility forecasting on daily equity data — into the high-frequency crypto setting, where microstructure noise, the Epps effect, and regime turbulence create a distinctly different environment.

---

## Scope (locked)

| Parameter | Choice |
|-----------|--------|
| Asset(s) | BTC/USDT (primary), ETH/USDT (extension) |
| Exchange | Binance |
| Base sampling | 1-minute bars |
| RV target horizon | Next 30 minutes |
| RV estimator | 5-minute sub-sampled RV (Yang–Zhang noise-robust) |
| Forecast models | HAR-RV (baseline), XGBoost (nonlinear baseline), HAR-RV + TDA, XGBoost + TDA |
| Evaluation | Walk-forward out-of-sample; QLIKE and MSE on log-RV |
| Data span | 2020-01-01 → 2023-12-31 (covers May 2021 crash, FTX Nov 2022) |

---

## Project Structure

```
tda-hf-crypto/
├── data/
│   ├── raw/          # Downloaded Binance OHLCV parquet files
│   └── processed/    # Cleaned RV targets + feature matrices
├── src/
│   ├── features/     # TDA feature pipeline (Takens embedding → persistence → vectorization)
│   ├── models/       # HAR-RV and XGBoost wrappers + walk-forward harness
│   └── utils/        # Data loading, RV estimation, logging
├── notebooks/        # EDA and results visualization
├── results/          # Forecast tables, regime breakdowns, significance tests
└── tests/
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Data collection** (downloads ~500MB, takes ~5 min):
```bash
python src/utils/fetch_data.py
```

**Build RV targets**:
```bash
python src/utils/build_rv.py
```

**Run baselines** (HAR-RV + XGBoost, walk-forward OOS):
```bash
python -m src.models.evaluation
```

---

## Data quality & cleaning

The realized-volatility target is sensitive to two opposite data artifacts, both
amplified by `binanceus` being far thinner than `binance.com` (used here only
because `binance.com` is geo-blocked in the US):

1. **Bad high/low ticks.** Single erroneous prints in the high/low column (e.g. a
   BTC high of \$138,070 while the bar trades at \$28,800) are invisible to a
   close-to-close return filter but feed straight into the Yang–Zhang estimator,
   inflating RV 100–1000×. Cleaned by clipping intrabar wicks beyond ±15% of the
   candle body (`cleaning.wick_threshold`, in `src/utils/cleaning.py`).
2. **Flat/illiquid bars.** ~50% of ETH 1-min bars have no trade (O=H=L=C), driving
   RV to exactly 0 → `log(0)` artifacts that dominate MSE-on-log. Handled with a
   variance noise floor (`realized_vol.rv_floor`, ≈3% annualised vol) applied to
   the target and all lags before logging.

Together these dropped baseline QLIKE from 2.88→0.64 (BTC) and MSE-on-log from
2.39→1.11 (ETH). **If you can source `binance.com`, Kraken, or Coinbase data,
prefer it** — the liquidity is materially better and would reduce reliance on the
noise floor. The walk-forward harness also purges the forward target horizon from
each training fold so training labels never overlap the OOS window.

> **Status:** HAR-RV and XGBoost baselines are implemented and validated. The TDA
> feature pipeline (`src/features/`) is the next milestone and is not yet built.

---

## Key References

- Souto, H.G. (2023). Topological Tail Dependence: Evidence from Forecasting Realized Volatility. *Journal of Finance and Data Science*. https://doi.org/10.1016/j.jfds.2023.100107
- Gidea, M. & Katz, Y. (2018). Topological Data Analysis of Financial Time Series. *Physica A*. https://doi.org/10.1016/j.physa.2018.04.039
- Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility. *Journal of Financial Econometrics*.
- Cont, R. (2001). Empirical Properties of Asset Returns. *Quantitative Finance*.
