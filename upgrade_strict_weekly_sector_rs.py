from pathlib import Path

PAGE = Path('pages/2_Final_Opportunities.py')


def main():
    text = PAGE.read_text(encoding='utf-8')
    if 'STRICT_WEEKLY_RS_V2' in text:
        print('strict weekly RS v2 already applied')
        return

    helper_anchor = "def technical_score(row):\n"
    if helper_anchor not in text:
        raise RuntimeError('technical_score anchor missing')

    helpers = r'''# STRICT_WEEKLY_RS_V2

def _weekly_close(ticker, period='5y'):
    try:
        d = yf.download(ticker, period=period, interval='1wk', auto_adjust=True, progress=False, timeout=25)
    except TypeError:
        d = yf.download(ticker, period=period, interval='1wk', auto_adjust=True, progress=False)
    except Exception:
        return pd.Series(dtype=float)
    if d is None or d.empty:
        return pd.Series(dtype=float)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    if 'Close' not in d.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(d['Close'], errors='coerce').dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def _recent_breakout(series, lookback=52, recent_weeks=8):
    s = pd.to_numeric(series, errors='coerce').dropna()
    if len(s) < lookback + recent_weeks + 2:
        return False, np.nan, np.nan, np.nan
    breakout_idx = None
    breakout_level = np.nan
    start = max(lookback, len(s) - recent_weeks)
    for i in range(start, len(s)):
        level = float(s.iloc[i-lookback:i].max())
        if float(s.iloc[i]) > level:
            breakout_idx = i
            breakout_level = level
    latest = float(s.iloc[-1])
    if breakout_idx is None:
        current_level = float(s.iloc[-lookback-1:-1].max())
        distance = (latest / current_level - 1) * 100 if current_level else np.nan
        return False, np.nan, current_level, distance
    weeks_ago = len(s) - 1 - breakout_idx
    still_above = latest >= breakout_level
    return bool(still_above), float(weeks_ago), float(breakout_level), (latest / breakout_level - 1) * 100


@st.cache_data(ttl=3600, show_spinner=False)
def strict_weekly_sector_rs():
    universe = load_official_nifty500()
    nifty = _weekly_close('^NSEI', '5y')
    if len(nifty) < 80:
        return universe, pd.DataFrame(), pd.DataFrame()

    stock_rows = []
    for _, r in universe.iterrows():
        c = _weekly_close(r['Ticker'], '5y')
        if len(c) < 80:
            continue
        x = pd.concat([c.rename('stock'), nifty.rename('nifty')], axis=1).dropna()
        if len(x) < 80:
            continue
        rs = (x['stock'] / x['nifty'])
        rs13 = (float(rs.iloc[-1] / rs.iloc[-14] - 1) * 100) if len(rs) > 13 else np.nan
        rs26 = (float(rs.iloc[-1] / rs.iloc[-27] - 1) * 100) if len(rs) > 26 else np.nan
        stock_rows.append({'Symbol': r['Symbol'], 'Company': r['Company Name'], 'Industry': r['Industry'], 'Ticker': r['Ticker'],
                           'RS 13W %': rs13, 'RS 26W %': rs26, 'Outperform 13W': bool(pd.notna(rs13) and rs13 > 0),
                           '_rs': rs})
    stocks = pd.DataFrame(stock_rows)
    if stocks.empty:
        return universe, stocks, pd.DataFrame()

    sector_rows = []
    for industry, g in stocks.groupby('Industry'):
        curves = []
        for rs in g['_rs']:
            z = rs.dropna()
            if len(z) < 80:
                continue
            z = z / float(z.iloc[0]) * 100.0
            curves.append(z.rename(str(len(curves))))
        if not curves:
            continue
        panel = pd.concat(curves, axis=1).sort_index().ffill().dropna(how='all')
        sector_rs = panel.median(axis=1).dropna()
        if len(sector_rs) < 80:
            continue
        br, weeks_ago, level, pct_above = _recent_breakout(sector_rs, 52, 8)
        rs13 = (float(sector_rs.iloc[-1] / sector_rs.iloc[-14] - 1) * 100) if len(sector_rs) > 13 else np.nan
        rs26 = (float(sector_rs.iloc[-1] / sector_rs.iloc[-27] - 1) * 100) if len(sector_rs) > 26 else np.nan
        breadth = 100 * g['Outperform 13W'].mean()
        strict = bool(br and pd.notna(rs13) and rs13 > 0 and pd.notna(rs26) and rs26 > 0 and breadth >= 50)
        score = float(np.clip(5 + (1.8 if br else 0) + np.tanh((rs13 if pd.notna(rs13) else 0)/8)*1.5 + np.tanh((rs26 if pd.notna(rs26) else 0)/15)*1.2 + (breadth-50)/50, 0, 10))
        sector_rows.append({'Industry': industry, 'Stocks': len(g), 'Strict Leader': strict,
                            'RS Breakout': br, 'Breakout Weeks Ago': weeks_ago, 'RS 13W %': rs13, 'RS 26W %': rs26,
                            'Outperform Breadth %': breadth, 'RS Above Breakout %': pct_above, 'Sector Score': score})
    sectors = pd.DataFrame(sector_rows)
    if not sectors.empty:
        sectors = sectors.sort_values(['Strict Leader','Sector Score','RS 13W %'], ascending=[False,False,False]).reset_index(drop=True)
    return universe, stocks.drop(columns=['_rs'], errors='ignore'), sectors


@st.cache_data(ttl=1800, show_spinner=False)
def stock_breakout_scan(ticker):
    w = _weekly_close(ticker, '10y')
    try:
        d = yf.download(ticker, period='10y', interval='1d', auto_adjust=True, progress=False, timeout=25)
    except TypeError:
        d = yf.download(ticker, period='10y', interval='1d', auto_adjust=True, progress=False)
    except Exception:
        d = pd.DataFrame()
    if isinstance(d, pd.DataFrame) and isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    m = pd.Series(dtype=float)
    if isinstance(d, pd.DataFrame) and not d.empty and 'Close' in d.columns:
        c = pd.to_numeric(d['Close'], errors='coerce').dropna()
        c.index = pd.to_datetime(c.index).tz_localize(None)
        m = c.resample('ME').last().dropna()

    wbr, wago, wlevel, wdist = _recent_breakout(w, 104, 8)
    mbr, mago, mlevel, mdist = _recent_breakout(m, 36, 3)
    weekly_status = '✅ RECENT 2Y WEEKLY BREAKOUT' if wbr else ('👀 NEAR 2Y WEEKLY BREAKOUT' if pd.notna(wdist) and -5 <= wdist <= 0 else 'Below weekly breakout zone')
    monthly_status = '✅ RECENT 3Y MONTHLY BREAKOUT' if mbr else ('👀 NEAR 3Y MONTHLY BREAKOUT' if pd.notna(mdist) and -5 <= mdist <= 0 else 'Below monthly breakout zone')
    return {'Weekly Status': weekly_status, 'Weekly Weeks Ago': wago, 'Weekly Level': wlevel, 'Weekly Distance %': wdist,
            'Monthly Status': monthly_status, 'Monthly Months Ago': mago, 'Monthly Level': mlevel, 'Monthly Distance %': mdist}


'''
    text = text.replace(helper_anchor, helpers + helper_anchor, 1)

    start = text.find('st.divider()\nst.subheader("🏅 Full NSE 500 Sector Leadership → Stock Analyzer")')
    end = text.find('if scan.empty:', start)
    if start == -1 or end == -1:
        raise RuntimeError('old full-sector UI block not found; run upgrade_full_nse_sector_dropdown.py first')

    ui = r'''st.divider()
st.subheader("🎯 Strict Weekly Sector RS Breakout → Stock Analyzer")
st.caption(
    "Leader rule: every NIFTY 500 industry is compared with NIFTY 50 on WEEKLY relative strength. "
    "A sector becomes a Strict Leader only when its sector/NIFTY50 RS line breaks its prior 52-week high now or within the last 8 weeks, "
    "RS is positive over both 13W and 26W, and at least 50% of its stocks are outperforming NIFTY 50 over 13 weeks."
)

try:
    with st.spinner("Comparing all NIFTY 500 sectors vs NIFTY 50 on weekly timeframe…"):
        rs_universe, rs_stocks, rs_sectors = strict_weekly_sector_rs()
except Exception as exc:
    rs_universe, rs_stocks, rs_sectors = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    st.warning(f"Strict weekly sector RS scan unavailable: {exc}")

if not rs_sectors.empty:
    strict = rs_sectors[rs_sectors['Strict Leader'] == True].copy()
    if strict.empty:
        st.warning("NO STRICT LEADER right now — no sector satisfies the complete weekly RS breakout rule. No sector is forced into leader status.")
        candidate_sector = rs_sectors.iloc[0]['Industry']
    else:
        candidate_sector = strict.iloc[0]['Industry']
        st.success(f"Current strict leader: **{candidate_sector}**")

    a1,a2,a3,a4 = st.columns(4)
    a1.metric('Strict Leaders', len(strict))
    a2.metric('Top Sector', candidate_sector)
    top_row = rs_sectors[rs_sectors['Industry']==candidate_sector].iloc[0]
    a3.metric('13W RS vs NIFTY', f"{top_row['RS 13W %']:+.1f}%")
    a4.metric('Breadth', f"{top_row['Outperform Breadth %']:.0f}%")

    st.dataframe(rs_sectors, use_container_width=True, hide_index=True,
        column_config={'Sector Score': st.column_config.ProgressColumn(min_value=0,max_value=10,format='%.1f'),
                       'Outperform Breadth %': st.column_config.ProgressColumn(min_value=0,max_value=100,format='%.0f%%')})

    selectable = strict['Industry'].tolist() if not strict.empty else rs_sectors['Industry'].head(5).tolist()
    chosen_sector = st.selectbox('Select sector for breakout stock scan', selectable, index=0)
    sector_stocks = rs_stocks[rs_stocks['Industry']==chosen_sector].copy()
    sector_stocks = sector_stocks.sort_values(['RS 13W %','RS 26W %'], ascending=False)
    if not sector_stocks.empty:
        labels = sector_stocks.apply(lambda r: f"{r['Symbol']} — {r['Company']}", axis=1).tolist()
        chosen_label = st.selectbox('Select stock', labels)
        chosen_symbol = chosen_label.split(' — ',1)[0]
        st.dataframe(sector_stocks[['Symbol','Company','RS 13W %','RS 26W %','Outperform 13W']], use_container_width=True, hide_index=True)
        if st.button(f'🔎 Scan weekly + monthly breakout — {chosen_symbol}', type='primary', use_container_width=True):
            with st.spinner(f'Scanning {chosen_symbol} breakout levels…'):
                br = stock_breakout_scan(chosen_symbol + '.NS')
                fs, assessed, pe = fundamental_quality(chosen_symbol + '.NS')
            st.session_state['strict_stock_breakout'] = (chosen_sector, chosen_symbol, br, fs, assessed, pe)
        saved = st.session_state.get('strict_stock_breakout')
        if saved and saved[0] == chosen_sector and saved[1] == chosen_symbol:
            _, _, br, fs, assessed, pe = saved
            b1,b2,b3,b4 = st.columns(4)
            b1.metric('Weekly breakout', br['Weekly Status'])
            b2.metric('Weekly distance', 'N/A' if pd.isna(br['Weekly Distance %']) else f"{br['Weekly Distance %']:+.1f}%")
            b3.metric('Monthly breakout', br['Monthly Status'])
            b4.metric('Monthly distance', 'N/A' if pd.isna(br['Monthly Distance %']) else f"{br['Monthly Distance %']:+.1f}%")
            st.write(f"**Weekly 2Y level:** {'N/A' if pd.isna(br['Weekly Level']) else f'₹{br[\"Weekly Level\"]:,.2f}'}  |  **Monthly 3Y level:** {'N/A' if pd.isna(br['Monthly Level']) else f'₹{br[\"Monthly Level\"]:,.2f}'}")
            st.write(f"**Fundamental score:** {'N/A' if pd.isna(fs) else f'{fs:.0f}%'} ({assessed} checks) · **Trailing P/E:** {'N/A' if pd.isna(pe) else f'{pe:.1f}'}")
            st.info('Priority setup: Strict Leader sector + stock outperforming NIFTY + stock near/recent weekly or monthly multiyear breakout. This is a research setup, not a profit guarantee.')

'''
    text = text[:start] + ui + text[end:]

    old_label = 'with st.expander("🔄 Sector leadership used in ranking", expanded=False):'
    if old_label in text:
        text = text.replace(old_label, 'with st.expander("🔄 26M scan-only sector context — NOT strict leader selector", expanded=False):', 1)

    PAGE.write_text(text, encoding='utf-8')
    print('strict weekly sector RS v2 applied')


if __name__ == '__main__':
    main()
