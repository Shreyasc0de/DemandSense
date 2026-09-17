import sys, unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from demandsense.data import (  # noqa: E402
    LEAKY_COLUMNS, WEATHER_REGRESSORS, DataError,
    check_spacing, clean, daily_demand, design_matrix, summarise,
)


def transactions(rows):
    """rows: (invoice, stockcode, quantity, date, price)"""
    return pd.DataFrame(
        [{"Invoice": i, "StockCode": s, "Quantity": q,
          "InvoiceDate": pd.Timestamp(d), "Price": p} for i, s, q, d, p in rows]
    )


BASE = [
    ("536365", "85123A", 6, "2009-12-01 08:26", 2.55),
    ("536366", "71053", 4, "2009-12-01 09:01", 3.39),
    ("536367", "84406B", 10, "2009-12-03 10:00", 2.75),
]


class TestClean(unittest.TestCase):
    def test_credit_notes_are_removed(self):
        df = transactions(BASE + [("C536500", "85123A", -6, "2009-12-04 10:00", 2.55)])
        kept, report = clean(df)
        self.assertEqual(report.rows_cancellations, 1)
        self.assertEqual(len(kept), 3)

    def test_a_return_cannot_cancel_an_earlier_sale(self):
        """The reason returns are dropped rather than netted off."""
        df = transactions(BASE + [("C900", "85123A", -6, "2010-03-01 10:00", 2.55)])
        kept, _ = clean(df)
        self.assertEqual(kept["Quantity"].sum(), 20)

    def test_non_positive_quantities_are_removed(self):
        df = transactions(BASE + [("536999", "85123A", 0, "2009-12-04 10:00", 2.55)])
        _, report = clean(df)
        self.assertEqual(report.rows_non_positive_quantity, 1)

    def test_rows_with_no_price_are_removed(self):
        df = transactions(BASE + [("536998", "85123A", 5, "2009-12-04 10:00", np.nan)])
        _, report = clean(df)
        self.assertEqual(report.rows_missing_price, 1)

    def test_postage_and_bank_charges_are_not_demand(self):
        df = transactions(BASE + [
            ("536997", "POST", 1, "2009-12-04 10:00", 18.0),
            ("536996", "BANK CHARGES", 1, "2009-12-04 10:00", 15.0),
        ])
        kept, report = clean(df)
        self.assertEqual(report.rows_non_product_codes, 2)
        self.assertEqual(len(kept), 3)

    def test_every_removed_row_is_counted_exactly_once(self):
        """The cleaning report has to add up, or the README cannot quote it."""
        df = transactions(BASE + [
            ("C536500", "85123A", -6, "2009-12-04 10:00", 2.55),
            ("536999", "85123A", 0, "2009-12-04 10:00", 2.55),
            ("536998", "85123A", 5, "2009-12-04 10:00", np.nan),
            ("536997", "POST", 1, "2009-12-04 10:00", 18.0),
        ])
        _, r = clean(df)
        self.assertEqual(
            r.rows_kept + r.rows_cancellations + r.rows_non_positive_quantity
            + r.rows_missing_price + r.rows_non_product_codes,
            r.rows_raw,
        )


class TestDailyDemand(unittest.TestCase):
    def test_days_with_no_trading_become_zeros_not_gaps(self):
        kept, _ = clean(transactions(BASE))
        daily = daily_demand(kept)
        # 1 Dec and 3 Dec traded; 2 Dec did not and must still be present.
        self.assertEqual(len(daily), 3)
        self.assertEqual(daily.loc[daily["ds"] == "2009-12-02", "y"].iloc[0], 0.0)

    def test_quantities_on_the_same_day_are_summed(self):
        kept, _ = clean(transactions(BASE))
        daily = daily_demand(kept)
        self.assertEqual(daily.loc[daily["ds"] == "2009-12-01", "y"].iloc[0], 10.0)

    def test_the_grid_is_strictly_daily(self):
        kept, _ = clean(transactions(BASE))
        check_spacing(daily_demand(kept))

    def test_intraday_timestamps_collapse_to_the_calendar_day(self):
        df = transactions([
            ("1", "A", 3, "2009-12-01 08:00", 1.0),
            ("2", "A", 4, "2009-12-01 23:59", 1.0),
        ])
        kept, _ = clean(df)
        daily = daily_demand(kept)
        self.assertEqual(len(daily), 1)
        self.assertEqual(daily["y"].iloc[0], 7.0)


class TestLeakage(unittest.TestCase):
    def test_the_target_cannot_be_a_regressor_under_any_of_its_names(self):
        df = pd.DataFrame({"ds": pd.date_range("2010-01-01", periods=3), "y": [1.0, 2, 3],
                           "temp_c": 5.0, "precip_mm": 0.0, "revenue": 10.0})
        for name in ("y", "revenue"):
            with self.assertRaises(AssertionError, msg=f"{name} was allowed through"):
                design_matrix(df, ("temp_c", name))

    def test_the_blocklist_records_a_reason_for_every_entry(self):
        for column, reason in LEAKY_COLUMNS.items():
            self.assertTrue(reason.strip(), f"{column} has no reason recorded")

    def test_a_regressor_that_is_not_in_the_series_is_an_error(self):
        df = pd.DataFrame({"ds": pd.date_range("2010-01-01", periods=3), "y": 1.0, "temp_c": 5.0})
        with self.assertRaises(AssertionError):
            design_matrix(df, ("temp_c", "humidity"))

    def test_the_weather_regressors_pass(self):
        df = pd.DataFrame({"ds": pd.date_range("2010-01-01", periods=3), "y": 1.0,
                           "temp_c": 5.0, "precip_mm": 0.0})
        out = design_matrix(df, WEATHER_REGRESSORS)
        self.assertEqual(list(out.columns), ["ds", "temp_c", "precip_mm"])


class TestSpacing(unittest.TestCase):
    def test_a_missing_day_is_rejected(self):
        df = pd.DataFrame({"ds": pd.to_datetime(["2010-01-01", "2010-01-02", "2010-01-04"])})
        with self.assertRaises(DataError):
            check_spacing(df)

    def test_a_duplicate_day_is_rejected(self):
        df = pd.DataFrame({"ds": pd.to_datetime(["2010-01-01", "2010-01-01"])})
        with self.assertRaises(DataError):
            check_spacing(df)


class TestSummary(unittest.TestCase):
    def test_zero_days_are_counted_and_excluded_from_the_trading_day_mean(self):
        ds = pd.date_range("2010-01-04", periods=7)  # Mon to Sun
        df = pd.DataFrame({"ds": ds, "y": [10.0, 10, 10, 10, 10, 0, 10],
                           "temp_c": np.linspace(1, 7, 7), "precip_mm": 0.0,
                           "dow": ds.dayofweek})
        facts = summarise(df)
        self.assertEqual(facts["n_zero_days"], 1)
        self.assertAlmostEqual(facts["mean_daily_units"], 60 / 7)
        self.assertAlmostEqual(facts["mean_units_on_trading_days"], 10.0)
        self.assertEqual(facts["n_saturdays"], 1)
        self.assertEqual(facts["n_saturdays_with_sales"], 0)


if __name__ == "__main__":
    unittest.main()
