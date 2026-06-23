"""
har_rv.py
---------
HAR-RV (Heterogeneous Autoregressive model of Realized Volatility).

Reference: Corsi (2009), "A Simple Approximate Long-Memory Model of Realized
Volatility", Journal of Financial Econometrics.

The standard HAR-RV model is:
    log_RV_{t+h} = c + β_S * log_RV_t^S + β_M * log_RV_t^M + β_L * log_RV_t^L + ε

where S/M/L are short, medium, and long backward-looking averages of log-RV.

In our high-frequency setting (5-min bars, 30-min forecast horizon):
    Short  : log_rv_lag_5min   (last 5 min  = 1 bar)
    Medium : log_rv_lag_60min  (last 60 min = 12 bars)
    Long   : log_rv_lag_480min (last 480 min = 96 bars ~ one trading session)

HAR-RV is deliberately simple — a linear OLS. Its strength is that it captures
long-memory volatility clustering remarkably well and is very hard to beat
out-of-sample. It is the standard benchmark in the volatility forecasting
literature.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# Features used by HAR-RV (must exist in the processed parquet)
HAR_FEATURES = [
    "log_rv_lag_5min",
    "log_rv_lag_60min",
    "log_rv_lag_480min",
]

TARGET = "rv_target"


class HARRV:
    """
    HAR-RV wrapper around sklearn LinearRegression.

    Fits on log(RV) lags, predicts log(RV) target, then exponentiates.
    Using log-RV makes the distribution more Gaussian and improves OLS fit.
    """

    def __init__(self, features: list[str] = HAR_FEATURES):
        self.features = features
        self.model = LinearRegression()
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "HARRV":
        X = df[self.features].values
        # Target: log of rv_target (clip to avoid log(0))
        y = np.log(df[TARGET].clip(lower=1e-12).values)
        self.model.fit(X, y)
        self.fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Returns predicted RV (not log-RV) — same units as rv_target."""
        assert self.fitted, "Call fit() first."
        X = df[self.features].values
        log_rv_pred = self.model.predict(X)
        return np.exp(log_rv_pred)

    def coef_summary(self) -> pd.Series:
        """Convenience: named coefficients for the paper's results table."""
        assert self.fitted
        names = ["intercept"] + self.features
        vals  = [self.model.intercept_] + list(self.model.coef_)
        return pd.Series(vals, index=names)
