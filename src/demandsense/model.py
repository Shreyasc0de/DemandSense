"""Forecasting engines, baselines, and rolling-origin evaluation.

Two engines are offered behind one interface. Prophet is the one the project
was built around. The gradient-boosting engine exists so the whole pipeline can
be run and tested in an environment without cmdstan, and so that Prophet has
something to be compared against rather than being assumed to be the right
choice.

Every engine returns a median and an 80% interval, because the app's stockout
logic needs a range and an interval nobody checks is decoration. ``coverage_80``
measures how often the interval actually contained the truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

INTERVAL_WIDTH = 0.80
LOWER_Q, UPPER_Q = 0.10, 0.90


def prophet_available() -> bool:
    try:
        import prophet  # noqa: F401
    except ImportError:
        return False
    return True


# ------------------------------------------------------------------ engines


@dataclass
class Forecast:
    """Point forecast plus an 80% interval, indexed by date."""

    ds: pd.Series
    yhat: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"ds": self.ds.values, "yhat": self.yhat, "lower": self.lower, "upper": self.upper}
        )


class ProphetEngine:
    """Prophet with optional external regressors."""

    name = "prophet"

    def __init__(self, regressors: tuple[str, ...] = ()):
        self.regressors = regressors
        self._model = None

    def fit(self, train: pd.DataFrame) -> "ProphetEngine":
        from prophet import Prophet

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,  # daily retail has a strong day-of-week shape
            daily_seasonality=False,
            interval_width=INTERVAL_WIDTH,
        )
        for regressor in self.regressors:
            model.add_regressor(regressor)
        model.fit(train[["ds", "y", *self.regressors]])
        self._model = model
        return self

    def predict(self, future: pd.DataFrame) -> Forecast:
        if self._model is None:
            raise RuntimeError("fit before predict")
        out = self._model.predict(future[["ds", *self.regressors]])
        return Forecast(future["ds"], out["yhat"].to_numpy(),
                        out["yhat_lower"].to_numpy(), out["yhat_upper"].to_numpy())

    def fit_predict(self, train: pd.DataFrame, future: pd.DataFrame) -> Forecast:
        return self.fit(train).predict(future)


class GradientBoostingEngine:
    """Quantile gradient boosting on calendar terms plus the same regressors.

    Three separate fits, at the 10th, 50th and 90th percentiles, so the
    interval comes from the model rather than from an assumed error
    distribution.
    """

    name = "gbdt"

    def __init__(self, regressors: tuple[str, ...] = (), seed: int = 7):
        self.regressors = regressors
        self.seed = seed
        self._fits: dict = {}
        self._origin = None

    def _features(self, df: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
        weeks = ((df["ds"] - origin).dt.days / 7.0).to_numpy()
        doy = df["ds"].dt.dayofyear.to_numpy()
        dow = df["ds"].dt.dayofweek.to_numpy()
        feats = {
            "weeks_since_origin": weeks,
            "sin_year": np.sin(2 * np.pi * doy / 365.25),
            "cos_year": np.cos(2 * np.pi * doy / 365.25),
            "sin_year2": np.sin(4 * np.pi * doy / 365.25),
            "cos_year2": np.cos(4 * np.pi * doy / 365.25),
            "week_of_year": df["ds"].dt.isocalendar().week.astype(float).to_numpy(),
            # Day of week matters enormously in daily retail, and this retailer
            # does not trade on Saturdays at all, so it is encoded explicitly
            # rather than left to a smooth seasonal term.
            "dow": dow.astype(float),
        }
        for d in range(7):
            feats[f"dow_{d}"] = (dow == d).astype(float)
        for regressor in self.regressors:
            feats[regressor] = df[regressor].astype(float).to_numpy()
        return pd.DataFrame(feats, index=df.index)

    def fit(self, train: pd.DataFrame) -> "GradientBoostingEngine":
        from sklearn.ensemble import GradientBoostingRegressor

        self._origin = train["ds"].min()
        x_train = self._features(train, self._origin)
        y = train["y"].to_numpy()
        self._fits = {}
        for label, alpha in (("lower", LOWER_Q), ("yhat", 0.5), ("upper", UPPER_Q)):
            gb = GradientBoostingRegressor(
                loss="quantile", alpha=alpha, n_estimators=300,
                max_depth=3, learning_rate=0.05, random_state=self.seed
            )
            gb.fit(x_train, y)
            self._fits[label] = gb
        return self

    def predict(self, future: pd.DataFrame) -> Forecast:
        if not self._fits:
            raise RuntimeError("fit before predict")
        x_future = self._features(future, self._origin)
        preds = {label: gb.predict(x_future) for label, gb in self._fits.items()}
        # Quantile fits are independent, so the three curves can cross. Sorting
        # them restores the ordering an interval has to satisfy.
        stacked = np.sort(np.vstack([preds["lower"], preds["yhat"], preds["upper"]]), axis=0)
        return Forecast(future["ds"], stacked[1], stacked[0], stacked[2])

    def fit_predict(self, train: pd.DataFrame, future: pd.DataFrame) -> Forecast:
        return self.fit(train).predict(future)


# ---------------------------------------------------------------- baselines


class SeasonalNaive:
    """The value one season earlier. The baseline any seasonal model must beat.

    For daily retail the season that matters is the week, so ``period`` is 7:
    predict next Tuesday with last Tuesday. That single line is a genuinely
    hard baseline and is the reason it is here.
    """

    name = "seasonal_naive"

    def __init__(self, period: int = 7):
        self.period = period

    def fit_predict(self, train: pd.DataFrame, future: pd.DataFrame) -> Forecast:
        history = train["y"].to_numpy()
        n_future = len(future)
        if len(history) < self.period:
            point = np.full(n_future, history.mean())
        else:
            # Walk the seasonal cycle forward, recycling if the horizon is long.
            tail = history[-self.period:]
            point = np.array([tail[i % self.period] for i in range(n_future)], dtype=float)
        # A naive method has no model interval; use the in-sample seasonal error.
        if len(history) > self.period:
            resid = history[self.period:] - history[: -self.period]
            spread = 1.2816 * resid.std(ddof=1)  # 80% interval under a normal error
        else:
            spread = 1.2816 * history.std(ddof=1)
        return Forecast(future["ds"], point, point - spread, point + spread)


class LastValue:
    """Carry the most recent observation forward. The floor for any method."""

    name = "last_value"

    def fit_predict(self, train: pd.DataFrame, future: pd.DataFrame) -> Forecast:
        history = train["y"].to_numpy()
        point = np.full(len(future), history[-1], dtype=float)
        spread = 1.2816 * np.diff(history).std(ddof=1) if len(history) > 2 else 0.0
        return Forecast(future["ds"], point, point - spread, point + spread)


class HistoricalMean:
    """Predict the training mean. Present so the error scale has a ceiling."""

    name = "mean"

    def fit_predict(self, train: pd.DataFrame, future: pd.DataFrame) -> Forecast:
        history = train["y"].to_numpy()
        point = np.full(len(future), history.mean(), dtype=float)
        spread = 1.2816 * history.std(ddof=1)
        return Forecast(future["ds"], point, point - spread, point + spread)


# ------------------------------------------------------------------ scoring


def score(actual: np.ndarray, forecast: Forecast) -> dict:
    """Point accuracy plus whether the 80% interval earns its name."""
    err = actual - forecast.yhat
    inside = (actual >= forecast.lower) & (actual <= forecast.upper)
    # MAPE is undefined where the actual is zero, so those weeks are excluded
    # rather than silently turned into infinity.
    nonzero = actual != 0
    mape = float(np.mean(np.abs(err[nonzero] / actual[nonzero])) * 100) if nonzero.any() else float("nan")
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": mape,
        "coverage_80": float(np.mean(inside)),
        "mean_interval_width": float(np.mean(forecast.upper - forecast.lower)),
        "n": int(len(actual)),
    }


@dataclass
class FoldResult:
    fold: int
    train_end: str
    horizon_start: str
    horizon_end: str
    metrics: dict = field(default_factory=dict)


def rolling_origin_folds(
    df: pd.DataFrame, horizon: int, initial: int, step: int
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window splits, ordered in time, no future rows in any training set."""
    folds = []
    end = initial
    while end + horizon <= len(df):
        folds.append((df.iloc[:end].copy(), df.iloc[end : end + horizon].copy()))
        end += step
    if not folds:
        raise ValueError(
            f"no folds: need at least initial + horizon = {initial + horizon} weeks, have {len(df)}"
        )
    return folds


def evaluate(
    df: pd.DataFrame, engine, horizon: int, initial: int, step: int
) -> tuple[list[FoldResult], dict]:
    """Run one engine across every fold and average the per-fold metrics."""
    results: list[FoldResult] = []
    for i, (train, test) in enumerate(rolling_origin_folds(df, horizon, initial, step), start=1):
        forecast = engine.fit_predict(train, test)
        results.append(
            FoldResult(
                fold=i,
                train_end=train["ds"].max().date().isoformat(),
                horizon_start=test["ds"].min().date().isoformat(),
                horizon_end=test["ds"].max().date().isoformat(),
                metrics=score(test["y"].to_numpy(), forecast),
            )
        )
    keys = ("mae", "rmse", "mape", "coverage_80", "mean_interval_width")
    summary = {k: float(np.mean([r.metrics[k] for r in results])) for k in keys}
    summary["n_folds"] = len(results)
    summary["sd_mae_across_folds"] = float(np.std([r.metrics["mae"] for r in results], ddof=1))
    return results, summary


def paired_comparison(folds_a: list[FoldResult], folds_b: list[FoldResult],
                      metric: str = "mae") -> dict:
    """Compare two engines fold by fold rather than by their averaged scores.

    Every engine sees identical folds, and folds differ enormously in
    difficulty: the hardest here is four times the easiest. Comparing two
    averaged MAEs throws that pairing away and leaves the fold-to-fold spread
    swamping the difference between methods. Differencing within each fold
    removes the shared difficulty, which is the whole reason the folds were
    held common.

    Positive ``mean_reduction`` means engine A beat engine B. The interval is
    normal-approximate on 13 folds, so read it as indicative rather than exact.
    """
    if len(folds_a) != len(folds_b):
        raise ValueError(f"fold counts differ: {len(folds_a)} and {len(folds_b)}")
    for a, b in zip(folds_a, folds_b, strict=True):
        if a.horizon_start != b.horizon_start:
            raise ValueError(f"folds are not aligned: {a.horizon_start} vs {b.horizon_start}")

    diffs = np.array([b.metrics[metric] - a.metrics[metric] for a, b in zip(folds_a, folds_b, strict=True)])
    n = len(diffs)
    mean = float(diffs.mean())
    se = float(diffs.std(ddof=1) / np.sqrt(n))
    return {
        "metric": metric,
        "n_folds": n,
        "mean_reduction": mean,
        "se": se,
        "ci_low": mean - 1.96 * se,
        "ci_high": mean + 1.96 * se,
        "folds_won": int((diffs > 0).sum()),
        # The interval straddling zero is the claim to report, not a p-value.
        "distinguishable": bool((mean - 1.96 * se) > 0 or (mean + 1.96 * se) < 0),
    }
