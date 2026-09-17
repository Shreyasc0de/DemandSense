"""Turn a forecast into the inventory decision, keeping the uncertainty attached.

The previous version of this project computed one stockout date from the point
forecast and displayed it as a critical alert. A forecast is a distribution, so
the honest output is a range: the interval that the model already produces is
carried through the cumulative sum instead of being discarded.

Higher demand empties the shelf sooner, so the UPPER demand bound gives the
EARLIEST stockout and the lower bound the latest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StockoutWindow:
    expected: pd.Timestamp | None
    earliest: pd.Timestamp | None
    latest: pd.Timestamp | None
    weeks_covered: int
    horizon_end: pd.Timestamp

    @property
    def certain(self) -> bool:
        """True when even the optimistic demand path runs the stock out."""
        return self.latest is not None

    @property
    def possible(self) -> bool:
        """True when any demand path inside the interval runs the stock out."""
        return self.earliest is not None


def _first_crossing(dates: pd.Series, demand: np.ndarray, stock: float) -> pd.Timestamp | None:
    depleted = stock - np.cumsum(demand)
    hit = np.flatnonzero(depleted < 0)
    return pd.Timestamp(dates.iloc[hit[0]]) if hit.size else None


def stockout_window(forecast: pd.DataFrame, current_stock: float) -> StockoutWindow:
    """When the stock runs out under the median, fast and slow demand paths.

    ``forecast`` needs ds, yhat, lower and upper, ordered in time.
    """
    if current_stock < 0:
        raise ValueError("current stock cannot be negative")
    for column in ("ds", "yhat", "lower", "upper"):
        if column not in forecast:
            raise KeyError(f"forecast is missing {column}")

    dates = forecast["ds"]
    # Demand cannot be negative even if a lower bound goes below zero.
    clip = lambda a: np.clip(a, 0, None)  # noqa: E731
    return StockoutWindow(
        expected=_first_crossing(dates, clip(forecast["yhat"].to_numpy()), current_stock),
        earliest=_first_crossing(dates, clip(forecast["upper"].to_numpy()), current_stock),
        latest=_first_crossing(dates, clip(forecast["lower"].to_numpy()), current_stock),
        weeks_covered=len(forecast),
        horizon_end=pd.Timestamp(dates.iloc[-1]),
    )


def reorder_by(window: StockoutWindow, lead_time_days: int) -> pd.Timestamp | None:
    """Latest safe order date: lead time before the EARLIEST plausible stockout.

    Planning against the expected date leaves you short half the time, which is
    the point of carrying the interval this far.
    """
    if window.earliest is None:
        return None
    return window.earliest - pd.Timedelta(days=lead_time_days)


def unmet_demand(forecast: pd.DataFrame, current_stock: float) -> float:
    """Units of median forecast demand falling after the stock is exhausted.

    This is an upper bound on lost sales, not an estimate of them. It assumes
    every customer who cannot buy is gone, when in practice some wait and some
    substitute. Nothing in this repository measures how many.
    """
    demand = np.clip(forecast["yhat"].to_numpy(), 0, None)
    shortfall = np.cumsum(demand) - current_stock
    return float(np.clip(shortfall[-1], 0, None)) if len(shortfall) else 0.0
