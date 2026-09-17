"""Backtest every method with rolling-origin CV and write the reports the README quotes.

Run:  python scripts/fetch_weather.py
      python scripts/train.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from demandsense import model as M  # noqa: E402
from demandsense.data import (  # noqa: E402
    WEATHER_REGRESSORS, build_series, check_spacing, design_matrix, summarise,
)

INK, ACCENT, WARN = "#2B2B2B", "#0B6E4F", "#C1443C"


def build_engines() -> list[tuple[str, object]]:
    """Every method the README may compare, baselines included."""
    engines: list[tuple[str, object]] = []
    if M.prophet_available():
        engines += [
            ("prophet_weather", M.ProphetEngine(regressors=WEATHER_REGRESSORS)),
            ("prophet_calendar_only", M.ProphetEngine(regressors=())),
        ]
    else:
        print("[train] prophet not importable; skipping both Prophet arms", file=sys.stderr)
    engines += [
        ("gbdt_weather", M.GradientBoostingEngine(regressors=WEATHER_REGRESSORS)),
        ("gbdt_calendar_only", M.GradientBoostingEngine(regressors=())),
        ("seasonal_naive_7d", M.SeasonalNaive(period=7)),
        ("last_value", M.LastValue()),
        ("mean", M.HistoricalMean()),
    ]
    return engines


def plot_backtest(df, engine, horizon, initial, step, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.plot(df["ds"], df["y"] / 1e3, color=INK, lw=0.8, alpha=0.75, label="Actual", zorder=3)
    for i, (train, test) in enumerate(M.rolling_origin_folds(df, horizon, initial, step)):
        fc = engine.fit_predict(train, test)
        ax.fill_between(fc.ds, fc.lower / 1e3, fc.upper / 1e3, color=ACCENT, alpha=0.20,
                        lw=0, label="80% interval" if i == 0 else None, zorder=2)
        ax.plot(fc.ds, fc.yhat / 1e3, color=ACCENT, lw=1.5,
                label="Forecast (held out)" if i == 0 else None, zorder=4)
    ax.set_ylabel("Units sold per day (thousands)")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_coverage(summaries: dict[str, dict], path: Path) -> None:
    names = sorted(summaries, key=lambda n: summaries[n]["coverage_80"])
    cover = [summaries[n]["coverage_80"] for n in names]
    colours = [ACCENT if abs(c - 0.80) <= 0.05 else WARN for c in cover]
    fig, ax = plt.subplots(figsize=(8.5, 0.46 * len(names) + 1.9))
    ax.barh(names, cover, color=colours, height=0.62)
    ax.axvline(0.80, color=INK, ls="--", lw=1.3)
    ax.annotate("nominal 80%", xy=(0.80, -0.85), xytext=(0.815, -0.85),
                color=INK, fontsize=9, va="center")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of held-out days inside the 80% interval")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_weekly_shape(facts: dict, path: Path) -> None:
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    means = [facts["dow_mean_units"][d] / 1e3 for d in range(7)]
    colours = [WARN if m == 0 else ACCENT for m in means]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(labels, means, color=colours, width=0.62)
    ax.set_ylabel("Mean units per day (thousands)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transactions", default=str(ROOT / "data" / "online_retail_II.xlsx"))
    parser.add_argument("--weather", default=str(ROOT / "data" / "weather_london.csv"))
    parser.add_argument("--horizon", type=int, default=28, help="forecast horizon in days")
    parser.add_argument("--initial", type=int, default=365, help="days in the first training window")
    parser.add_argument("--step", type=int, default=28, help="days between fold origins")
    args = parser.parse_args()

    df, cleaning = build_series(args.transactions, args.weather)
    check_spacing(df)
    design_matrix(df, WEATHER_REGRESSORS)  # raises if the target reaches the features

    facts = summarise(df)
    print(f"[train] {facts['n_days']} days, {facts['first_day']} to {facts['last_day']}; "
          f"{facts['n_zero_days']} with no sales")
    print(f"[train] kept {cleaning.rows_kept:,} of {cleaning.rows_raw:,} transaction rows")

    reports = ROOT / "reports"
    (reports / "figures").mkdir(parents=True, exist_ok=True)

    engines = build_engines()
    summaries: dict[str, dict] = {}
    all_folds: dict[str, list] = {}
    fold_rows: list[dict] = []
    for name, engine in engines:
        folds, summary = M.evaluate(df, engine, args.horizon, args.initial, args.step)
        summaries[name] = summary
        all_folds[name] = folds
        for f in folds:
            fold_rows.append({"engine": name, "fold": f.fold, "train_end": f.train_end,
                              "horizon_start": f.horizon_start, "horizon_end": f.horizon_end,
                              **f.metrics})
        print(f"[train] {name:24} MAE {summary['mae']:>10,.0f}  MAPE {summary['mape']:>7.2f}%  "
              f"coverage {summary['coverage_80']:.2f}")

    best = min(summaries, key=lambda k: summaries[k]["mae"])
    baselines = ("seasonal_naive_7d", "last_value", "mean")
    best_baseline = min((b for b in baselines if b in summaries),
                        key=lambda k: summaries[k]["mae"])

    metrics = {
        "source": {
            "transactions": "UCI Online Retail II (CC BY 4.0)",
            "weather": "Open-Meteo historical reanalysis, London (CC BY 4.0)",
        },
        "horizon_days": args.horizon,
        "initial_days": args.initial,
        "step_days": args.step,
        "prophet_available": M.prophet_available(),
        "cleaning": cleaning.as_dict(),
        "series": facts,
        "engines": summaries,
        "best_by_mae": best,
        "best_baseline": best_baseline,
        "best_vs_baseline_mae_pct": float(
            100 * (summaries[best_baseline]["mae"] - summaries[best]["mae"])
            / summaries[best_baseline]["mae"]
        ),
    }

    # Paired fold-by-fold comparisons. Folds are shared across engines, so the
    # difference within each fold is the comparison with any power; averaged
    # MAEs alone cannot separate these methods.
    comparisons: dict[str, dict] = {}
    pairs = [(f"{fam}_weather", f"{fam}_calendar_only") for fam in ("prophet", "gbdt")]
    pairs += [(best, best_baseline)] if best != best_baseline else []
    for a, b in pairs:
        if a in all_folds and b in all_folds and a != b:
            comparisons[f"{a}__vs__{b}"] = M.paired_comparison(all_folds[a], all_folds[b])
    metrics["paired_comparisons"] = comparisons

    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(fold_rows).to_csv(reports / "folds.csv", index=False)
    df.to_csv(reports / "series.csv", index=False)

    best_engine = dict(engines)[best]
    plot_backtest(df, best_engine, args.horizon, args.initial, args.step,
                  reports / "figures" / "backtest.png")
    plot_coverage(summaries, reports / "figures" / "coverage.png")
    plot_weekly_shape(facts, reports / "figures" / "weekly_shape.png")

    # Fit the winner on the whole series and save it, so the app loads a model
    # rather than refitting on page load. An app that refits is not the model
    # whose backtest is published here.
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(
        {"engine": best_engine.fit(df), "engine_name": best,
         "regressors": WEATHER_REGRESSORS,
         "last_observed_day": df["ds"].max().date().isoformat()},
        models_dir / "forecaster.joblib",
    )

    for label, c in comparisons.items():
        verdict = "distinguishable" if c["distinguishable"] else "NOT distinguishable"
        print(f"[train] {label}: {c['mean_reduction']:+,.0f} MAE "
              f"[{c['ci_low']:+,.0f}, {c['ci_high']:+,.0f}], "
              f"wins {c['folds_won']}/{c['n_folds']} folds, {verdict}")
    print(f"[train] best by MAE: {best} ({metrics['best_vs_baseline_mae_pct']:+.1f}% vs {best_baseline})")
    print(f"[train] wrote reports/metrics.json, folds.csv, series.csv and 3 figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
