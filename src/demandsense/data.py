"""Load the Online Retail II transactions and join them to London weather.

Two real sources, both redistributable with attribution:

* UCI Online Retail II: every transaction of a UK online gift retailer from
  2009-12-01 to 2011-12-09, roughly a million rows.
* Open-Meteo historical reanalysis: daily temperature and precipitation for
  London over the same span.

Everything downstream is a documented reduction of those two files. The
cleaning decisions that could move a number are recorded in the returned
``CleaningReport`` rather than buried, so the README can quote how many rows
each one removed and the test suite can check them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: The retailer is in the UK and ships mostly domestically, so one weather
#: station stands in for the whole customer base. This is an approximation and
#: the README says so.
LONDON_LAT, LONDON_LON = 51.5074, -0.1278

TARGET = "y"

#: Regressors a planner either knows or supplies as a scenario. Anything else
#: from the transaction file describes the day being forecast and would not be
#: available when the forecast is made.
WEATHER_REGRESSORS = ("temp_c", "precip_mm")

#: Columns that are the target by another name, or arithmetic on it.
LEAKY_COLUMNS: dict[str, str] = {
    "y": "the target",
    "Quantity": "the target before aggregation",
    "revenue": "Quantity times Price, so it carries the target",
    "n_invoices": "counted from the same rows as the target",
}


class DataError(RuntimeError):
    """Raised when an input file is missing or does not look like the real one."""


@dataclass
class CleaningReport:
    rows_raw: int
    rows_cancellations: int
    rows_non_positive_quantity: int
    rows_missing_price: int
    rows_non_product_codes: int
    rows_kept: int
    non_product_codes_seen: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


#: StockCodes that are not products. Postage, bank charges, manual adjustments
#: and the like move with billing rather than with demand.
NON_PRODUCT_CODES = (
    "POST", "DOT", "C2", "M", "m", "BANK CHARGES", "S", "AMAZONFEE",
    "B", "CRUK", "PADS", "D", "ADJUST", "ADJUST2", "TEST001", "TEST002",
)


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Read both sheets of the Online Retail II workbook into one frame."""
    path = Path(path)
    if not path.exists():
        raise DataError(
            f"{path} not found. Download online_retail_II.xlsx from "
            "https://archive.ics.uci.edu/dataset/502/online+retail+ii"
        )
    sheets = pd.read_excel(path, sheet_name=None)
    frames = []
    for name, sheet in sheets.items():
        sheet = sheet.copy()
        sheet["sheet"] = name
        frames.append(sheet)
    df = pd.concat(frames, ignore_index=True)

    df.columns = [str(c).strip() for c in df.columns]
    required = {"Invoice", "StockCode", "Quantity", "InvoiceDate", "Price"}
    missing = required - set(df.columns)
    if missing:
        raise DataError(f"{path} is missing columns {sorted(missing)}; is this the right file?")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Drop what is not a completed sale of a product, counting each removal.

    Four exclusions, each with a reason:

    * Credit notes (Invoice starts with C) are returns, a separate process from
      demand. Keeping them would let a January refund cancel a December sale.
    * Non-positive quantities are the same thing recorded differently.
    * Rows with no price are not completed sales.
    * Non-product StockCodes are postage, bank charges and manual adjustments.
    """
    rows_raw = len(df)
    invoice = df["Invoice"].astype(str)
    code = df["StockCode"].astype(str).str.strip()

    is_cancellation = invoice.str.upper().str.startswith("C")
    is_non_positive = df["Quantity"] <= 0
    is_missing_price = df["Price"].isna() | (df["Price"] <= 0)
    is_non_product = code.str.upper().isin({c.upper() for c in NON_PRODUCT_CODES})

    keep = ~(is_cancellation | is_non_positive | is_missing_price | is_non_product)
    report = CleaningReport(
        rows_raw=rows_raw,
        rows_cancellations=int(is_cancellation.sum()),
        rows_non_positive_quantity=int((is_non_positive & ~is_cancellation).sum()),
        rows_missing_price=int((is_missing_price & ~is_cancellation & ~is_non_positive).sum()),
        rows_non_product_codes=int(
            (is_non_product & ~is_cancellation & ~is_non_positive & ~is_missing_price).sum()
        ),
        rows_kept=int(keep.sum()),
        non_product_codes_seen=tuple(sorted(set(code[is_non_product].str.upper()))),
    )
    return df.loc[keep].copy(), report


def daily_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Total units sold per calendar day, on a complete daily grid.

    Days the retailer did not trade are kept as zeros rather than dropped. The
    gaps are a real feature of this business (it does not invoice on Saturdays),
    and a model that never sees them cannot learn the weekly shape.
    """
    daily = (
        df.assign(date=df["InvoiceDate"].dt.normalize())
        .groupby("date", as_index=False)["Quantity"]
        .sum()
        .rename(columns={"date": "ds", "Quantity": TARGET})
    )
    grid = pd.DataFrame({"ds": pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")})
    out = grid.merge(daily, on="ds", how="left")
    out[TARGET] = out[TARGET].fillna(0.0).astype(float)
    return out


def load_weather(path: str | Path) -> pd.DataFrame:
    """Read the Open-Meteo CSV that scripts/fetch_weather.py writes."""
    path = Path(path)
    if not path.exists():
        raise DataError(f"{path} not found. Run `python scripts/fetch_weather.py` first.")
    weather = pd.read_csv(path, parse_dates=["ds"])
    missing = {"ds", *WEATHER_REGRESSORS} - set(weather.columns)
    if missing:
        raise DataError(f"{path} is missing columns {sorted(missing)}")
    return weather


def build_series(transactions_path: str | Path, weather_path: str | Path) -> tuple[pd.DataFrame, CleaningReport]:
    """The one series everything downstream uses, plus the cleaning audit."""
    raw = load_transactions(transactions_path)
    kept, report = clean(raw)
    demand = daily_demand(kept)
    weather = load_weather(weather_path)

    df = demand.merge(weather[["ds", *WEATHER_REGRESSORS]], on="ds", how="left", validate="one_to_one")
    if df[list(WEATHER_REGRESSORS)].isna().any().any():
        gap = df.loc[df[list(WEATHER_REGRESSORS)].isna().any(axis=1), "ds"]
        raise DataError(
            f"weather is missing for {len(gap)} day(s), first {gap.iloc[0].date()}. "
            "Re-run scripts/fetch_weather.py covering the full span of the sales data."
        )
    df["dow"] = df["ds"].dt.dayofweek
    return df, report


def check_spacing(df: pd.DataFrame) -> None:
    """Assert a complete daily grid: no gaps, no duplicates."""
    gaps = df["ds"].diff().dropna().dt.days.unique()
    if len(gaps) != 1 or gaps[0] != 1:
        raise DataError(f"expected a strict 1-day grid, found gaps of {sorted(gaps)} days")
    if df["ds"].duplicated().any():
        raise DataError("duplicate dates in the series")


def design_matrix(df: pd.DataFrame, regressors: tuple[str, ...]) -> pd.DataFrame:
    """Feature frame, with a loud failure if the target reaches it by any name."""
    overlap = set(regressors) & set(LEAKY_COLUMNS)
    if overlap:
        raise AssertionError(
            "leakage: "
            + "; ".join(f"{c} is {LEAKY_COLUMNS[c]}" for c in sorted(overlap))
        )
    absent = set(regressors) - set(df.columns)
    if absent:
        raise AssertionError(f"regressors not present in the series: {sorted(absent)}")
    return df[["ds", *regressors]].copy()


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    """Pearson correlation that returns NaN instead of warning on a flat input."""
    if len(a) < 2 or a.std(ddof=1) == 0 or b.std(ddof=1) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def summarise(df: pd.DataFrame) -> dict:
    """Facts about the series the README is allowed to quote."""
    trading = df[df[TARGET] > 0]
    by_dow = df.groupby("dow")[TARGET].mean()
    return {
        "n_days": int(len(df)),
        "first_day": df["ds"].min().date().isoformat(),
        "last_day": df["ds"].max().date().isoformat(),
        "n_zero_days": int((df[TARGET] == 0).sum()),
        "zero_day_share": float((df[TARGET] == 0).mean()),
        "n_saturdays": int((df["dow"] == 5).sum()),
        "n_saturdays_with_sales": int(((df["dow"] == 5) & (df[TARGET] > 0)).sum()),
        "mean_daily_units": float(df[TARGET].mean()),
        "mean_units_on_trading_days": float(trading[TARGET].mean()),
        "sd_daily_units": float(df[TARGET].std(ddof=1)),
        "max_daily_units": float(df[TARGET].max()),
        "busiest_day": df.loc[df[TARGET].idxmax(), "ds"].date().isoformat(),
        "mean_temp_c": float(df["temp_c"].mean()),
        "min_temp_c": float(df["temp_c"].min()),
        "max_temp_c": float(df["temp_c"].max()),
        "corr_temp_units": _safe_corr(df["temp_c"], df[TARGET]),
        "corr_temp_units_trading_days": _safe_corr(trading["temp_c"], trading[TARGET]),
        "dow_mean_units": {int(k): float(v) for k, v in by_dow.items()},
    }
