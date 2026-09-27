"""Shared macro scoring and data-quality rules used across dashboard pages."""
from __future__ import annotations

import numpy as np
import pandas as pd


# Direction is the preferred direction of the 3-month move for Indian risk assets.
MACRO_RULES = (
    ("NIFTY 50", 1, 2.0),
    ("India VIX", -1, 1.5),
    ("USD/INR", -1, 1.5),
    ("Brent Crude", -1, 1.5),
    ("Dollar Index", -1, 1.0),
    ("US 10Y Yield", -1, 1.0),
    ("Gold", 1, 0.5),
)
TOTAL_WEIGHT = sum(rule[2] for rule in MACRO_RULES)
MIN_COVERAGE = 70
MIN_AVAILABLE_DRIVERS = 5
MAX_STALENESS_DAYS = 5


def completed_monthly_bars(monthly: pd.DataFrame, *, now=None) -> pd.DataFrame:
    """Keep completed calendar months; drop the developing current-month bar."""
    if monthly is None or monthly.empty:
        return pd.DataFrame() if monthly is None else monthly.copy()
    today = (pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()
             if now is None else pd.Timestamp(now).tz_localize(None).normalize())
    months = pd.DatetimeIndex(pd.to_datetime(monthly.index)).to_period("M")
    return monthly.loc[months < today.to_period("M")].copy()


def score_macro(frame: pd.DataFrame, *, min_coverage: int = MIN_COVERAGE):
    """Return score (0–100), regime, coverage, attribution, stale flag and as-of.

    Missing drivers are excluded from the score and coverage is explicit. Below
    the coverage threshold the function withholds both score and regime.
    """
    if frame is None or frame.empty:
        return np.nan, "DATA INCOMPLETE", 0, pd.DataFrame(), True, pd.NaT

    name_col = "Indicator" if "Indicator" in frame.columns else "Driver"
    value_col = "3M %" if "3M %" in frame.columns else None
    if value_col is None:
        return np.nan, "DATA INCOMPLETE", 0, pd.DataFrame(), True, pd.NaT

    latest_by_name = {}
    for _, row in frame.iterrows():
        name = str(row.get(name_col, ""))
        try:
            latest_by_name[name] = float(row[value_col])
        except (TypeError, ValueError):
            latest_by_name[name] = np.nan

    available_weight = 0.0
    weighted_signal = 0.0
    contributions = []
    for name, direction, weight in MACRO_RULES:
        move = latest_by_name.get(name, np.nan)
        valid = np.isfinite(move)
        contribution = float(np.tanh(direction * move / 5.0)) if valid else np.nan
        if valid:
            available_weight += weight
            weighted_signal += contribution * weight
        contributions.append({"Driver": name, "3M %": move, "Contribution": contribution,
                              "Weight": weight, "Available": bool(valid)})

    coverage = int(round(100 * available_weight / TOTAL_WEIGHT))
    available_drivers = sum(bool(item["Available"]) for item in contributions)
    attribution = pd.DataFrame(contributions)
    dates = pd.to_datetime(frame.get("As Of", pd.Series(dtype=object)), errors="coerce").dropna()
    newest = dates.max() if not dates.empty else pd.NaT
    now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()
    # One fresh feed must not conceal another stale feed that still contributes.
    stale = (
        dates.empty
        or any((now - pd.Timestamp(value).tz_localize(None).normalize()).days > MAX_STALENESS_DAYS
               for value in dates)
    )

    critical_missing = [name for name in ("NIFTY 50", "India VIX", "USD/INR")
                        if not np.isfinite(latest_by_name.get(name, np.nan))]
    if coverage < min_coverage or available_drivers < MIN_AVAILABLE_DRIVERS or stale or critical_missing:
        reason = ("LOW COVERAGE" if coverage < min_coverage or available_drivers < MIN_AVAILABLE_DRIVERS else
                  "MISSING KEY INPUTS" if critical_missing else "STALE DATA")
        return np.nan, f"DATA INCOMPLETE — {reason}", coverage, attribution, bool(stale), newest

    score = float(np.clip(50 + 50 * weighted_signal / available_weight, 0, 100))
    values = latest_by_name
    nifty = values.get("NIFTY 50", np.nan)
    crude = values.get("Brent Crude", np.nan)
    fx = values.get("USD/INR", np.nan)
    vix_rows = frame[frame[name_col].astype(str) == "India VIX"]
    vix = pd.to_numeric(vix_rows.get("Latest", pd.Series(dtype=float)), errors="coerce").dropna()
    vix_level = float(vix.iloc[-1]) if len(vix) else np.nan
    inflation_stress = int(np.isfinite(crude) and crude > 10) + int(np.isfinite(fx) and fx > 3)
    market_stress = int(np.isfinite(nifty) and nifty < -5) + int(np.isfinite(vix_level) and vix_level > 20)
    if score >= 76 and (not np.isfinite(nifty) or nifty > 0) and market_stress == 0:
        regime = "EARLY / RISK-ON"
    elif score >= 68 and market_stress == 0:
        regime = "MID CYCLE / EXPANSION"
    elif inflation_stress >= 1 and score >= 38:
        regime = "LATE CYCLE / INFLATION-SENSITIVE"
    elif score < 38 or market_stress >= 2:
        regime = "RISK-OFF / CONTRACTION"
    else:
        regime = "MID-TO-LATE / MIXED"
    return score, regime, coverage, attribution, False, newest
