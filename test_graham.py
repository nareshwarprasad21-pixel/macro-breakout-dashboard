import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from graham import _ten_year_tests, _twenty_year_dividends, nse_ticker


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
