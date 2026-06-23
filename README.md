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

---

## Key References

- Souto, H.G. (2023). Topological Tail Dependence: Evidence from Forecasting Realized Volatility. *Journal of Finance and Data Science*. https://doi.org/10.1016/j.jfds.2023.100107
- Gidea, M. & Katz, Y. (2018). Topological Data Analysis of Financial Time Series. *Physica A*. https://doi.org/10.1016/j.physa.2018.04.039
- Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility. *Journal of Financial Econometrics*.
- Cont, R. (2001). Empirical Properties of Asset Returns. *Quantitative Finance*.
