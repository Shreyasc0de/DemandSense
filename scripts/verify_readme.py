"""Re-derive every figure quoted in README.md from reports/ and fail on drift.

The README makes claims about a specific run. Those claims are only worth
something if they still match the data, so this script recomputes each one from
`reports/metrics.json` and `reports/folds.csv` rather than trusting the prose.

Paired comparisons are recomputed from the raw per-fold errors and then
cross-checked against what `train.py` recorded, so a change to
`paired_comparison()` that silently alters a published conclusion also fails
here.

Tolerance on a quoted figure is half a unit of its last printed digit. The
quoted value is passed as the string the README prints, not as a number, so
that trailing zeros survive: "0.880" earns a tolerance of 0.0005, while the
number 0.880 would arrive as "0.88" and silently claim one ten times looser.

Run:  python scripts/verify_readme.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from demandsense.model import FoldResult, paired_comparison  # noqa: E402

REPORTS = ROOT / "reports"
checks = 0
failures: list[str] = []


def near(label: str, actual: float, quoted: str) -> None:
    """Agreement to half a unit of the last digit the README prints."""
    global checks
    checks += 1
    dot = quoted.replace(",", "").find(".")
    text = quoted.replace(",", "")
    decimals = 0 if dot == -1 else len(text) - dot - 1
    tolerance = 0.5 * 10**-decimals
    expected = float(text)
    if not abs(actual - expected) <= tolerance:
        failures.append(
            f"{label}: README says {quoted}, reports give {actual:,.6g} "
            f"(tolerance {tolerance:g})"
        )


def exact(label: str, actual, quoted) -> None:
    global checks
    checks += 1
    if actual != quoted:
        failures.append(f"{label}: README says {quoted!r}, reports give {actual!r}")


def identical(label: str, actual: float, expected: float, note: str) -> None:
    """Agreement to floating-point noise, for two numbers that share a source."""
    global checks
    checks += 1
    if abs(actual - expected) > 1e-6:
        failures.append(f"{label}: {note} ({actual!r} vs {expected!r})")


# ----------------------------------------------------------------- the reports

for required in ("metrics.json", "folds.csv"):
    if not (REPORTS / required).exists():
        sys.exit(f"missing reports/{required}. Run `python scripts/train.py` first.")

metrics = json.loads((REPORTS / "metrics.json").read_text())
folds = pd.read_csv(REPORTS / "folds.csv")
series, cleaning, engines = metrics["series"], metrics["cleaning"], metrics["engines"]


def fold_results(engine: str) -> list[FoldResult]:
    rows = folds[folds["engine"] == engine].sort_values("fold")
    if rows.empty:
        raise SystemExit(f"folds.csv has no rows for {engine}")
    return [
        FoldResult(int(r["fold"]), r["train_end"], r["horizon_start"],
                   r["horizon_end"], {"mae": float(r["mae"])})
        for _, r in rows.iterrows()
    ]


# ------------------------------------------------------------- the sources

exact("transactions source", metrics["source"]["transactions"],
      "UCI Online Retail II (CC BY 4.0)")
exact("weather source", metrics["source"]["weather"],
      "Open-Meteo historical reanalysis, London (CC BY 4.0)")

# ------------------------------------------------------------- the series

exact("days in the series", series["n_days"], 739)
exact("first day", series["first_day"], "2009-12-01")
exact("last day", series["last_day"], "2011-12-09")
exact("days with no sales", series["n_zero_days"], 135)
near("share of days with no sales", 100 * series["zero_day_share"], "18.3")
exact("Saturdays in the series", series["n_saturdays"], 105)
exact("Saturdays with any sales", series["n_saturdays_with_sales"], 1)
near("busiest day units", series["max_daily_units"], "125,513")
exact("busiest day", series["busiest_day"], "2010-09-27")
near("correlation, temperature and units", series["corr_temp_units"], "-0.032")
near("correlation on trading days only",
     series["corr_temp_units_trading_days"], "-0.084")

dow_quoted = {"0": "19,562", "1": "20,463", "2": "19,353", "3": "22,397",
              "4": "16,003", "5": "49", "6": "10,026"}
day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
for key, quoted in dow_quoted.items():
    near(f"mean units on {day_names[int(key)]}", series["dow_mean_units"][key], quoted)

# The weekday/Saturday contrast the README states in words.
weekday_mean = sum(series["dow_mean_units"][str(d)] for d in range(5)) / 5
near("weekday mean, quoted as roughly 20,000", weekday_mean / 1000, "19.6")

# ------------------------------------------------------------ the cleaning

near("raw transaction rows", cleaning["rows_raw"], "1,067,371")
near("rows kept", cleaning["rows_kept"], "1,037,081")
near("credit notes removed", cleaning["rows_cancellations"], "19,494")
near("non-product codes removed", cleaning["rows_non_product_codes"], "4,589")
near("non-positive quantities removed", cleaning["rows_non_positive_quantity"], "3,457")
near("rows with no price removed", cleaning["rows_missing_price"], "2,750")

identical(
    "cleaning counts sum to the raw total",
    cleaning["rows_kept"] + cleaning["rows_cancellations"]
    + cleaning["rows_non_positive_quantity"] + cleaning["rows_missing_price"]
    + cleaning["rows_non_product_codes"],
    cleaning["rows_raw"],
    "the exclusion counts no longer add up, so the cleaning table is wrong",
)
for code in ("POST", "BANK CHARGES", "AMAZONFEE", "ADJUST"):
    exact(f"{code} is named as a non-product code",
          code in cleaning["non_product_codes_seen"], True)

# ---------------------------------------------------------- the fold design

exact("forecast horizon in days", metrics["horizon_days"], 28)
exact("initial training window in days", metrics["initial_days"], 365)
exact("step between fold origins in days", metrics["step_days"], 28)
exact("Prophet arms are present in this run", metrics["prophet_available"], True)
for name, summary in engines.items():
    exact(f"{name} ran 13 folds", summary["n_folds"], 13)

# --------------------------------------------------------- the results table

table = {
    "gbdt_weather": ("4,917", "37.7", "0.885"),
    "gbdt_calendar_only": ("5,226", "39.0", "0.879"),
    "prophet_weather": ("5,982", "41.2", "0.901"),
    "prophet_calendar_only": ("5,983", "41.9", "0.904"),
    "seasonal_naive_7d": ("6,479", "49.6", "0.934"),
    "mean": ("8,920", "53.0", "0.920"),
    "last_value": ("10,226", "72.0", "0.901"),
}
exact("the table covers every engine that ran", set(table), set(engines))
for name, (mae, mape, coverage) in table.items():
    near(f"{name} MAE", engines[name]["mae"], mae)
    near(f"{name} MAPE", engines[name]["mape"], mape)
    near(f"{name} coverage", engines[name]["coverage_80"], coverage)

# Every interval is wider than nominal, which is the claim the section makes.
for name, summary in engines.items():
    exact(f"{name} 80% interval over-covers", summary["coverage_80"] > 0.80, True)
near("narrowest coverage in the table",
     min(s["coverage_80"] for s in engines.values()), "0.879")
near("widest coverage in the table",
     max(s["coverage_80"] for s in engines.values()), "0.934")

# --------------------------------------------------- the paired comparisons

quoted_pairs = {
    ("prophet_weather", "prophet_calendar_only"): ("2", "-81", "84", 6, False),
    ("gbdt_weather", "gbdt_calendar_only"): ("309", "-278", "895", 5, False),
    ("gbdt_weather", "seasonal_naive_7d"): ("1,562", "690", "2,434", 11, True),
    ("prophet_weather", "seasonal_naive_7d"): ("498", "-525", "1,521", 6, False),
    ("gbdt_weather", "prophet_weather"): ("1,064", "230", "1,899", 11, True),
}
for (a, b), (mean, low, high, won, distinguishable) in quoted_pairs.items():
    result = paired_comparison(fold_results(a), fold_results(b))
    near(f"{a} vs {b}, MAE reduction", result["mean_reduction"], mean)
    near(f"{a} vs {b}, CI lower", result["ci_low"], low)
    near(f"{a} vs {b}, CI upper", result["ci_high"], high)
    exact(f"{a} vs {b}, folds won", result["folds_won"], won)
    exact(f"{a} vs {b}, distinguishable from zero",
          result["distinguishable"], distinguishable)

# What train.py recorded must match what this script recomputes.
for label, recorded in metrics.get("paired_comparisons", {}).items():
    a, b = label.split("__vs__")
    fresh = paired_comparison(fold_results(a), fold_results(b))
    identical(f"recorded vs recomputed: {label}", fresh["mean_reduction"],
              recorded["mean_reduction"],
              "metrics.json disagrees with a fresh computation from folds.csv")

# Prophet's weather gain as a share of its own error, quoted as 0.03%.
prophet_gain = paired_comparison(
    fold_results("prophet_weather"), fold_results("prophet_calendar_only")
)["mean_reduction"]
near("Prophet's weather gain as a percentage of its MAE",
     100 * prophet_gain / engines["prophet_weather"]["mae"], "0.03")

# The fold-difficulty spread that motivates pairing.
winner = folds[folds["engine"] == "gbdt_weather"]["mae"]
near("hardest fold for the winning model", winner.max(), "12,675")
near("easiest fold for the winning model", winner.min(), "2,308")
exact("hardest fold is about five times the easiest",
      round(winner.max() / winner.min()) == 5, True)

# ------------------------------------------------------------------ verdict

if failures:
    print(f"README verification FAILED: {len(failures)} of {checks} figures drifted\n")
    for failure in failures:
        print(f"  - {failure}")
    raise SystemExit(1)

print(f"all {checks} README figures match the generated reports")
