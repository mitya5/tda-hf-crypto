"""
xgboost_baseline.py
-------------------
XGBoost baseline for realized-volatility forecasting.

Uses the same HAR-RV lag features as the linear baseline but lets XGBoost
learn nonlinear interactions. This is important because Souto (2023) found
that TDA features improved *nonlinear* models significantly more than linear
ones — so we need this baseline to make the same comparison in our setting.

Hyperparameters are intentionally conservative (shallow trees, moderate
learning rate) to avoid overfitting on a single walk-forward fold. We do
not tune them aggressively; the goal is a credible nonlinear baseline, not
a maximally-optimized model.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


# Same features as HAR-RV for a fair head-to-head comparison.
# The TDA-augmented version will add extra columns on top of these.
HAR_FEATURES = [
    "log_rv_lag_5min",
    "log_rv_lag_60min",
    "log_rv_lag_480min",
]

TARGET = "rv_target"

DEFAULT_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,   # regularise — prevents fitting tiny clusters
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)


class XGBBaseline:
    """
    XGBoost wrapper that mirrors the HARRV interface (fit / predict).

    Fits on log(RV) lags, predicts log(RV), exponentiates output — same
    convention as HARRV so the evaluation harness can treat them identically.
    """

    def __init__(self, features: list[str] = HAR_FEATURES,
                 params: dict | None = None):
        self.features = features
        p = DEFAULT_PARAMS.copy()
        if params:
            p.update(params)
        self.model = XGBRegressor(**p)
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "XGBBaseline":
        X = df[self.features].values
        y = np.log(df[TARGET].clip(lower=1e-12).values)
        self.model.fit(X, y)
        self.fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        assert self.fitted, "Call fit() first."
        X = df[self.features].values
        return np.exp(self.model.predict(X))

    def feature_importance(self) -> pd.Series:
        assert self.fitted
        return pd.Series(
            self.model.feature_importances_,
            index=self.features
        ).sort_values(ascending=False)
