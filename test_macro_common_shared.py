import unittest
import numpy as np
import pandas as pd
from macro_common import completed_monthly_bars, score_macro


def macro_frame(as_of=None):
    names = ["NIFTY 50", "India VIX", "USD/INR", "Brent Crude", "Dollar Index", "US 10Y Yield", "Gold"]
    return pd.DataFrame({"Driver": names, "Latest": [25000,14,83,75,100,4.2,2500],
        "3M %": [2,-3,1,5,1,2,4],
        "As Of": [as_of or pd.Timestamp.now(tz="Asia/Kolkata").date().isoformat()] * 7})


class MacroCommonTests(unittest.TestCase):
    def test_shared_macro_score_and_coverage(self):
        score, regime, coverage, attribution, stale, _ = score_macro(macro_frame())
        self.assertTrue(np.isfinite(score)); self.assertGreaterEqual(score,0); self.assertLessEqual(score,100)
        self.assertEqual(coverage,100); self.assertFalse(stale); self.assertEqual(len(attribution),7)
        self.assertNotIn("DATA INCOMPLETE", regime)

    def test_missing_coverage_and_key_feed_withhold_score(self):
        score, regime, coverage, *_ = score_macro(macro_frame().iloc[:4])
        self.assertTrue(np.isnan(score)); self.assertEqual(coverage,72); self.assertIn("LOW COVERAGE",regime)
        frame=macro_frame(); frame.loc[frame.Driver=="India VIX","3M %"]=np.nan
        score, regime, coverage, *_=score_macro(frame)
        self.assertTrue(np.isnan(score)); self.assertGreaterEqual(coverage,70); self.assertIn("MISSING KEY INPUTS",regime)

    def test_stale_contributing_feed_withholds_score(self):
        frame=macro_frame(); frame.loc[frame.Driver=="Gold","As Of"]=(pd.Timestamp.now(tz="Asia/Kolkata")-pd.Timedelta(days=20)).date().isoformat()
        score,regime,*_=score_macro(frame)
        self.assertTrue(np.isnan(score)); self.assertIn("STALE DATA",regime)

    def test_month_filter_drops_only_developing_bar(self):
        dates=pd.to_datetime(["2026-07-31","2026-08-31","2026-09-30"])
        bars=pd.DataFrame({"Close":[10,11,12]},index=dates)
        result=completed_monthly_bars(bars,now="2026-09-27")
        self.assertEqual(list(result.index),list(dates[:2]))


if __name__ == "__main__":
    unittest.main()
