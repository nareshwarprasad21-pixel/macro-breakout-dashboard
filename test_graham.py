import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from graham import _historical_eps_growth_rate, _ten_year_tests, _twenty_year_dividends, nse_ticker


class GrahamCalculationsTest(unittest.TestCase):
    def test_nse_symbol_normalization(self):
        self.assertEqual(nse_ticker("bse"), "BSE.NS")
        self.assertEqual(nse_ticker("BSE.NS.NS"), "BSE.NS")
        with self.assertRaises(ValueError):
            nse_ticker("BSE.BO")

    def test_ten_year_eps_requires_complete_history(self):
        dates = pd.to_datetime([f"{year}-03-31" for year in range(2016, 2026)])
        eps = pd.Series(np.arange(10, 20), index=dates)
        self.assertEqual(_ten_year_tests(eps), (True, True, 10))
        positive, growth, years = _ten_year_tests(eps.drop(eps.index[4]))
        self.assertIsNone(positive)
        self.assertIsNone(growth)
        self.assertEqual(years, 9)

    def test_historical_eps_growth_proxy_uses_first_and_last_three_year_averages(self):
        dates = pd.to_datetime([f"{year}-03-31" for year in range(2016, 2026)])
        eps = pd.Series([10, 10, 10, 11, 12, 13, 14, 20, 20, 20], index=dates)
        # First 3-year average = 10, last 3-year average = 20; average-window midpoints are 7 years apart.
        expected = (20.0 / 10.0) ** (1.0 / 7.0) - 1.0
        self.assertAlmostEqual(_historical_eps_growth_rate(eps), expected, places=12)
        self.assertTrue(np.isnan(_historical_eps_growth_rate(eps.drop(eps.index[4]))))

    def test_margin_of_safety_formula(self):
        graham_number = 1000.0
        cmp = 800.0
        margin = (graham_number - cmp) / graham_number * 100.0
        self.assertAlmostEqual(margin, 20.0)

    def test_growth_formula_arithmetic(self):
        eps = 20.0
        g = 10.0
        y = 5.5
        value = eps * (8.5 + 2.0 * g) * 4.4 / y
        self.assertAlmostEqual(value, 456.0)

    def test_dividend_test_excludes_incomplete_current_year(self):
        dates = pd.to_datetime([f"{year}-06-30" for year in range(2005, 2026)], utc=True)
        dividends = pd.Series(1.0, index=dates)
        now = datetime(2026, 8, 15, tzinfo=timezone.utc)
        self.assertTrue(_twenty_year_dividends(dividends, now))
        self.assertFalse(_twenty_year_dividends(dividends.drop(pd.Timestamp("2015-06-30", tz="UTC")), now))

    def test_truncated_dividend_history_is_unavailable(self):
        dates = pd.to_datetime([f"{year}-06-30" for year in range(2010, 2026)], utc=True)
        self.assertIsNone(_twenty_year_dividends(pd.Series(1.0, index=dates),
                                                datetime(2026, 8, 15, tzinfo=timezone.utc)))


if __name__ == "__main__":
    unittest.main()
