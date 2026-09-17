import sys, unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from demandsense.inventory import reorder_by, stockout_window, unmet_demand  # noqa: E402


def forecast(yhat, spread=20.0, n=None):
    n = n or len(yhat)
    ds = pd.date_range("2012-11-02", periods=n, freq="7D")
    yhat = np.asarray(yhat, dtype=float)
    return pd.DataFrame({"ds": ds, "yhat": yhat, "lower": yhat - spread, "upper": yhat + spread})


class TestStockoutWindow(unittest.TestCase):
    def test_faster_demand_empties_the_shelf_sooner(self):
        w = stockout_window(forecast([100] * 10, spread=50), current_stock=450)
        self.assertLess(w.earliest, w.expected)
        self.assertLess(w.expected, w.latest)

    def test_expected_date_is_the_first_week_cumulative_demand_passes_the_stock(self):
        w = stockout_window(forecast([100, 100, 100, 100], spread=0), current_stock=250)
        # 100, 200, 300: the third week is the first above 250.
        self.assertEqual(w.expected, pd.Timestamp("2012-11-16"))

    def test_ample_stock_produces_no_dates_at_all(self):
        w = stockout_window(forecast([10] * 10, spread=1), current_stock=10_000)
        self.assertIsNone(w.expected)
        self.assertIsNone(w.earliest)
        self.assertFalse(w.possible)
        self.assertFalse(w.certain)

    def test_a_stockout_only_the_fast_path_reaches_is_possible_but_not_certain(self):
        # Median demand totals 400 over 4 weeks; the upper path totals 600.
        w = stockout_window(forecast([100] * 4, spread=50), current_stock=450)
        self.assertTrue(w.possible)
        self.assertFalse(w.certain)
        self.assertIsNone(w.expected)

    def test_negative_lower_bounds_do_not_create_negative_demand(self):
        w = stockout_window(forecast([10, 10, 10], spread=1000), current_stock=25)
        self.assertIsNone(w.latest)  # the slow path consumes nothing, so it never runs out

    def test_missing_interval_columns_are_rejected(self):
        with self.assertRaises(KeyError):
            stockout_window(pd.DataFrame({"ds": [], "yhat": []}), 100)

    def test_negative_stock_is_rejected(self):
        with self.assertRaises(ValueError):
            stockout_window(forecast([10] * 3), current_stock=-1)


class TestReorder(unittest.TestCase):
    def test_reorder_is_lead_time_before_the_earliest_stockout_not_the_expected_one(self):
        w = stockout_window(forecast([100] * 10, spread=50), current_stock=450)
        self.assertEqual(reorder_by(w, lead_time_days=7), w.earliest - pd.Timedelta(days=7))
        self.assertLess(reorder_by(w, 7), w.expected)

    def test_no_stockout_means_no_reorder_date(self):
        w = stockout_window(forecast([10] * 5), current_stock=1_000_000)
        self.assertIsNone(reorder_by(w, 7))


class TestUnmetDemand(unittest.TestCase):
    def test_unmet_demand_is_total_demand_minus_stock(self):
        self.assertAlmostEqual(unmet_demand(forecast([100] * 4, spread=0), 250), 150.0)

    def test_sufficient_stock_leaves_nothing_unmet(self):
        self.assertAlmostEqual(unmet_demand(forecast([100] * 4, spread=0), 1000), 0.0)


if __name__ == "__main__":
    unittest.main()
