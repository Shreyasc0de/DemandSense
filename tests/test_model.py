import sys, unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from demandsense import model as M  # noqa: E402


def series(n=143, seed=0):
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2010-02-05", periods=n, freq="7D")
    y = 1e6 + 2000 * np.arange(n) + 1e5 * np.sin(2 * np.pi * ds.dayofyear / 365.25) + rng.normal(0, 4e4, n)
    return pd.DataFrame({
        "ds": ds, "y": y,
        "Temperature": 60 + 20 * np.sin(2 * np.pi * ds.dayofyear / 365.25),
        "IsHoliday": 0,
    })


class TestFolds(unittest.TestCase):
    def test_no_training_row_is_dated_after_its_test_rows(self):
        """The property that separates a time-series split from a random one."""
        df = series()
        for train, test in M.rolling_origin_folds(df, horizon=13, initial=91, step=13):
            self.assertLess(train["ds"].max(), test["ds"].min())

    def test_training_windows_expand(self):
        df = series()
        folds = M.rolling_origin_folds(df, horizon=13, initial=91, step=13)
        sizes = [len(train) for train, _ in folds]
        self.assertEqual(sizes, sorted(sizes))
        self.assertEqual(sizes[0], 91)

    def test_every_horizon_is_the_requested_length(self):
        df = series()
        for _, test in M.rolling_origin_folds(df, horizon=13, initial=91, step=13):
            self.assertEqual(len(test), 13)

    def test_too_short_a_series_is_an_error_not_an_empty_list(self):
        with self.assertRaises(ValueError):
            M.rolling_origin_folds(series(n=20), horizon=13, initial=91, step=13)


class TestScoring(unittest.TestCase):
    def test_coverage_counts_points_inside_the_interval(self):
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        fc = M.Forecast(pd.Series(pd.date_range("2020-01-01", periods=4)),
                        yhat=np.array([1.0, 2.0, 3.0, 4.0]),
                        lower=np.array([0.0, 0.0, 0.0, 10.0]),
                        upper=np.array([2.0, 2.0, 2.5, 20.0]))
        s = M.score(actual, fc)
        # Inside, inside, outside (3 > 2.5), outside (4 < 10).
        self.assertAlmostEqual(s["coverage_80"], 0.5)
        self.assertAlmostEqual(s["mae"], 0.0)

    def test_mae_and_rmse_differ_when_errors_are_uneven(self):
        actual = np.array([0.0, 0.0])
        fc = M.Forecast(pd.Series(pd.date_range("2020-01-01", periods=2)),
                        yhat=np.array([0.0, 10.0]), lower=np.zeros(2), upper=np.zeros(2))
        s = M.score(actual, fc)
        self.assertAlmostEqual(s["mae"], 5.0)
        self.assertAlmostEqual(s["rmse"], np.sqrt(50.0))


class TestBaselines(unittest.TestCase):
    def test_seasonal_naive_returns_the_value_52_weeks_earlier(self):
        df = series(n=120)
        train, test = df.iloc[:104], df.iloc[104:108]
        fc = M.SeasonalNaive(period=52).fit_predict(train, test)
        expected = train["y"].to_numpy()[-52:][:4]
        np.testing.assert_allclose(fc.yhat, expected)

    def test_last_value_is_flat(self):
        df = series(n=60)
        fc = M.LastValue().fit_predict(df.iloc[:50], df.iloc[50:55])
        self.assertEqual(len(set(np.round(fc.yhat, 9))), 1)
        self.assertAlmostEqual(fc.yhat[0], df["y"].iloc[49])

    def test_every_engine_returns_an_ordered_interval(self):
        df = series()
        train, test = df.iloc[:91], df.iloc[91:104]
        for engine in (M.GradientBoostingEngine(regressors=("Temperature",)),
                       M.SeasonalNaive(), M.LastValue(), M.HistoricalMean()):
            fc = engine.fit_predict(train, test)
            self.assertTrue((fc.lower <= fc.yhat).all(), engine.name)
            self.assertTrue((fc.yhat <= fc.upper).all(), engine.name)


class TestInSampleIsOptimistic(unittest.TestCase):
    def test_scoring_on_the_training_rows_flatters_the_model(self):
        """The defect in the previous version of this project, as a test.

        Fitting and then scoring on the same rows reports an error far below
        what the same model achieves on weeks it has not seen. Any MAE quoted
        without a held-out split is measuring memorisation.
        """
        df = series()
        train, test = df.iloc[:91].copy(), df.iloc[91:104].copy()
        engine = M.GradientBoostingEngine(regressors=("Temperature",))

        in_sample = M.score(train["y"].to_numpy(), engine.fit_predict(train, train))["mae"]
        held_out = M.score(test["y"].to_numpy(), engine.fit_predict(train, test))["mae"]

        self.assertLess(in_sample, held_out,
                        "in-sample error should be the smaller, flattering one")
        self.assertGreater(held_out / in_sample, 1.5,
                           "the gap should be large enough to change a conclusion")


if __name__ == "__main__":
    unittest.main()


class TestFitPredictSplit(unittest.TestCase):
    """The app fits once and predicts many times as the scenario slider moves."""

    def test_separate_fit_then_predict_matches_fit_predict(self):
        df = series()
        train, test = df.iloc[:91], df.iloc[91:104]
        for engine_a, engine_b in (
            (M.GradientBoostingEngine(regressors=("Temperature",)),
             M.GradientBoostingEngine(regressors=("Temperature",))),
        ):
            combined = engine_a.fit_predict(train, test)
            split = engine_b.fit(train).predict(test)
            np.testing.assert_allclose(combined.yhat, split.yhat)
            np.testing.assert_allclose(combined.lower, split.lower)
            np.testing.assert_allclose(combined.upper, split.upper)

    def test_predict_before_fit_is_an_error(self):
        with self.assertRaises(RuntimeError):
            M.GradientBoostingEngine().predict(series().iloc[:5])

    def test_refitting_replaces_state_rather_than_accumulating_it(self):
        df = series()
        engine = M.GradientBoostingEngine(regressors=("Temperature",))
        engine.fit(df.iloc[:91])
        first = engine.predict(df.iloc[91:104]).yhat.copy()
        engine.fit(df.iloc[:120])
        second = engine.predict(df.iloc[91:104]).yhat
        self.assertFalse(np.allclose(first, second),
                         "a refit on more data should change the forecast")


class TestPairedComparison(unittest.TestCase):
    """Folds are shared across engines, so the comparison must use the pairing."""

    @staticmethod
    def folds(maes):
        return [M.FoldResult(fold=i + 1, train_end="2011-01-01",
                             horizon_start=f"2011-01-{i + 2:02d}", horizon_end="2011-02-01",
                             metrics={"mae": m}) for i, m in enumerate(maes)]

    def test_a_consistent_small_edge_is_detected_despite_huge_fold_spread(self):
        """The reason pairing matters: fold difficulty swamps the method difference."""
        hard = [12000, 6000, 3000, 5000, 3000, 5000, 3500, 4000, 4000, 5000, 4000, 5000, 6000]
        a = self.folds([h - 400 for h in hard])
        b = self.folds(hard)
        result = M.paired_comparison(a, b)
        self.assertAlmostEqual(result["mean_reduction"], 400.0)
        self.assertTrue(result["distinguishable"])
        self.assertEqual(result["folds_won"], 13)
        # Unpaired, the fold spread alone is thousands, far larger than the edge.
        spread = np.std(hard, ddof=1)
        self.assertGreater(spread, 400 * 4)

    def test_an_inconsistent_edge_is_not_called_distinguishable(self):
        a = self.folds([5000, 4000, 6000, 3000, 5500, 4200, 3800, 5100, 4700, 3900, 5300, 4400, 4800])
        b = self.folds([4800, 4300, 5700, 3400, 5200, 4600, 3500, 5400, 4400, 4300, 5000, 4700, 4500])
        result = M.paired_comparison(a, b)
        self.assertFalse(result["distinguishable"])
        self.assertLess(result["ci_low"], 0)
        self.assertGreater(result["ci_high"], 0)

    def test_the_sign_convention_is_a_beats_b(self):
        better = self.folds([100] * 5)
        worse = self.folds([200] * 5)
        self.assertGreater(M.paired_comparison(better, worse)["mean_reduction"], 0)
        self.assertLess(M.paired_comparison(worse, better)["mean_reduction"], 0)

    def test_misaligned_folds_are_rejected_rather_than_silently_compared(self):
        a = self.folds([100, 200, 300])
        b = self.folds([100, 200, 300])
        b[1].horizon_start = "2099-01-01"
        with self.assertRaises(ValueError):
            M.paired_comparison(a, b)

    def test_unequal_fold_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            M.paired_comparison(self.folds([1, 2, 3]), self.folds([1, 2]))
