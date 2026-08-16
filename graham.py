"""Data acquisition and calculations for the Graham scorecard.

The module deliberately represents missing observations with NaN/None.  In
particular, an absent debt, dividend, or EPS record is never interpreted as
zero.  This file has no Streamlit dependency so the calculations can be unit
tested without starting the dashboard.
"""

from datetime import datetime, timezone
import re

import numpy as np
import pandas as pd
import requests
import yfinance as yf


def nse_ticker(value):
    """Return a canonical Yahoo NSE ticker, without ever creating `.NS.NS`."""
    symbol = str(value).strip().upper()
    symbol = re.sub(r"(?:\.NS)+$", "", symbol)
    if not symbol or "." in symbol:
        raise ValueError("Enter an NSE symbol (for example BSE or BSE.NS).")
    return f"{symbol}.NS"


def _statement_row(frame, names):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.Series(dtype=float)
    labels = {str(label).strip().lower(): label for label in frame.index}
    for name in names:
        if name.lower() in labels:
            result = pd.to_numeric(frame.loc[labels[name.lower()]], errors="coerce").dropna()
            result.index = pd.to_datetime(result.index, errors="coerce")
            return result[result.index.notna()].sort_index()
    return pd.Series(dtype=float)


def _latest(series):
    return float(series.iloc[-1]) if isinstance(series, pd.Series) and not series.empty else np.nan


def _annual_yahoo_series(symbol, field, session=requests):
    """Fetch up to 12 reported annual observations from Yahoo's time-series API."""
    now = int(datetime.now(timezone.utc).timestamp())
    url = f"https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{symbol}"
    response = session.get(
        url,
        params={"symbol": symbol, "type": f"annual{field}", "period1": now - 13 * 366 * 86400,
                "period2": now, "limit": 20},
        headers={"User-Agent": "Mozilla/5.0"}, timeout=15,
    )
    response.raise_for_status()
    blocks = response.json().get("timeseries", {}).get("result", [])
    observations = {}
    for block in blocks:
        for item in block.get(f"annual{field}", []):
            raw = item.get("reportedValue", {}).get("raw")
            date = pd.to_datetime(item.get("asOfDate"), errors="coerce")
            if pd.notna(date) and raw is not None:
                observations[date] = float(raw)
    return pd.Series(observations, dtype=float).sort_index()


def _ten_year_tests(eps):
    """Evaluate only a complete sequence of ten distinct annual EPS reports."""
    if not isinstance(eps, pd.Series) or eps.empty:
        return None, None, 0
    eps = pd.to_numeric(eps, errors="coerce").dropna()
    eps.index = pd.to_datetime(eps.index, errors="coerce")
    eps = eps[eps.index.notna()].groupby(eps.index.year).last().sort_index()
    if len(eps) < 10:
        return None, None, len(eps)
    last = eps.iloc[-10:]
    # Gaps are missing reports, not zero earnings.
    if list(last.index) != list(range(int(last.index[-1]) - 9, int(last.index[-1]) + 1)):
        return None, None, len(eps)
    positive = bool((last > 0).all())
    first_avg, last_avg = last.iloc[:3].mean(), last.iloc[-3:].mean()
    growth = bool(last_avg >= 1.33 * first_avg) if first_avg > 0 else False
    return positive, growth, len(eps)


def _twenty_year_dividends(dividends, now=None):
    """Test 20 *completed* calendar years when Yahoo history covers the window."""
    if not isinstance(dividends, pd.Series) or dividends.empty:
        return None
    now = now or datetime.now(timezone.utc)
    values = pd.to_numeric(dividends, errors="coerce").dropna()
    index = pd.to_datetime(values.index, errors="coerce", utc=True)
    values.index = index
    values = values[values.index.notna()]
    end = now.year - 1
    start = end - 19
    # An event at/before the first required year establishes event-history
    # coverage. Without it, zero-event years cannot be distinguished from a
    # truncated Yahoo response and the honest result is N/A.
    if values.empty or values.index.year.min() > start:
        return None
    annual = values.groupby(values.index.year).sum().reindex(range(start, end + 1))
    return bool((annual.notna() & (annual > 0)).all())


def fetch_graham_data(ticker):
    """Fetch live/reported inputs and calculate the ten Graham criteria."""
    symbol = nse_ticker(ticker)
    stock = yf.Ticker(symbol)
    try:
        info = stock.info or {}
    except Exception:
        info = {}

    def frame(*attributes):
        for attribute in attributes:
            try:
                value = getattr(stock, attribute)
                value = value() if callable(value) else value
                if isinstance(value, pd.DataFrame) and not value.empty:
                    return value.copy()
            except Exception:
                continue
        return pd.DataFrame()

    income = frame("income_stmt", "financials", "get_income_stmt")
    balance = frame("balance_sheet", "get_balance_sheet")
    revenue = _statement_row(income, ["Total Revenue", "Operating Revenue"])
    eps_statement = _statement_row(income, ["Diluted EPS", "Basic EPS"])
    current_assets = _statement_row(balance, ["Current Assets", "Total Current Assets"])
    current_liabilities = _statement_row(balance, ["Current Liabilities", "Total Current Liabilities"])
    debt = _statement_row(balance, ["Long Term Debt And Capital Lease Obligation", "Long Term Debt",
                                    "Long Term Debt Noncurrent"])
    equity = _statement_row(balance, ["Stockholders Equity", "Total Stockholder Equity",
                                      "Common Stock Equity"])
    statement_shares = _statement_row(balance, ["Ordinary Shares Number", "Share Issued"])

    def number(value):
        try:
            result = float(value)
            return result if np.isfinite(result) else np.nan
        except (TypeError, ValueError):
            return np.nan

    price = number(info.get("currentPrice", info.get("regularMarketPrice")))
    if pd.isna(price):
        try:
            price = number(stock.fast_info["last_price"])
        except Exception:
            pass
    if pd.isna(price):
        try:
            history = stock.history(period="5d", interval="1d", auto_adjust=False)
            price = _latest(pd.to_numeric(history.get("Close"), errors="coerce").dropna())
        except Exception:
            pass

    sales = _latest(revenue)
    if pd.isna(sales):
        sales = number(info.get("totalRevenue"))
    ca, cl = _latest(current_assets), _latest(current_liabilities)
    current_ratio = ca / cl if pd.notna(ca) and pd.notna(cl) and cl > 0 else number(info.get("currentRatio"))
    nwc = ca - cl if pd.notna(ca) and pd.notna(cl) else np.nan
    long_debt = _latest(debt)
    if pd.isna(long_debt):
        long_debt = number(info.get("longTermDebt"))

    try:
        eps_history = _annual_yahoo_series(symbol, "DilutedEPS")
        if eps_history.empty:
            eps_history = _annual_yahoo_series(symbol, "BasicEPS")
    except Exception:
        eps_history = pd.Series(dtype=float)
    # Statement data remains a valid fallback, but does not magically satisfy
    # a ten-year requirement when Yahoo returns only four statements.
    eps_for_history = eps_history if not eps_history.empty else eps_statement
    positive10, growth10, eps_years = _ten_year_tests(eps_for_history)
    eps_for_average = eps_history if len(eps_history) >= 3 else eps_statement
    avg3_eps = float(eps_for_average.iloc[-3:].mean()) if len(eps_for_average) >= 3 else np.nan

    bvps = number(info.get("bookValue"))
    shares = number(info.get("sharesOutstanding"))
    if pd.isna(shares):
        try:
            shares = number(stock.fast_info["shares"])
        except Exception:
            shares = _latest(statement_shares)
    if pd.isna(bvps) and pd.notna(_latest(equity)) and pd.notna(shares) and shares > 0:
        bvps = _latest(equity) / shares

    try:
        dividends = stock.dividends
    except Exception:
        dividends = pd.Series(dtype=float)
    dividend20 = _twenty_year_dividends(dividends)

    pe3 = price / avg3_eps if pd.notna(price) and pd.notna(avg3_eps) and avg3_eps > 0 else np.nan
    pb = price / bvps if pd.notna(price) and pd.notna(bvps) and bvps > 0 else np.nan
    combined = pe3 * pb if pd.notna(pe3) and pd.notna(pb) else np.nan
    graham_number = np.sqrt(22.5 * avg3_eps * bvps) if pd.notna(avg3_eps) and avg3_eps > 0 and pd.notna(bvps) and bvps > 0 else np.nan

    return {
        "symbol": symbol, "company": info.get("longName") or info.get("shortName") or symbol[:-3],
        "price": price, "sales": sales, "current_ratio": current_ratio, "nwc": nwc,
        "long_debt": long_debt, "eps_positive_10y": positive10, "dividend20": dividend20,
        "eps_growth_10y": growth10, "avg3_eps": avg3_eps, "pe3": pe3, "bvps": bvps,
        "pb": pb, "combined": combined, "graham_no": graham_number, "eps_years": eps_years,
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "sources": "Yahoo Finance live quote, annual financial statements, fundamentals time-series and dividend events",
    }
