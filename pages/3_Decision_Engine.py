from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title='Decision Engine', page_icon='🎯', layout='wide')


def text_metric(container, label, value):
    """Responsive text card for long categorical values; avoids Streamlit metric ellipsis."""
    safe_label = str(label).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    safe_value = str(value).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    container.markdown(
        f"""
        <div style="min-height:118px;padding:16px 18px;border:1px solid rgba(148,163,184,.18);
                    border-radius:16px;background:linear-gradient(145deg,rgba(19,31,54,.96),rgba(10,20,38,.96));
                    box-shadow:0 10px 28px rgba(0,0,0,.16);display:flex;flex-direction:column;justify-content:center;">
          <div style="color:#cbd5e1;font-size:.86rem;font-weight:650;margin-bottom:9px;">{safe_label}</div>
          <div style="color:#f8fafc;font-size:clamp(1.18rem,1.55vw,1.75rem);font-weight:500;
                      line-height:1.16;white-space:normal;overflow-wrap:anywhere;word-break:normal;">{safe_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.markdown('''<style>.block-container{max-width:1650px;padding-top:1.5rem}[data-testid="stMetric"]{border:1px solid rgba(128,128,128,.22);border-radius:12px;padding:10px}[data-testid="stMetricValue"]{white-space:normal;font-size:1.55rem}@media(max-width:900px){.block-container{padding:1rem .6rem!important}[data-testid="stMetricValue"]{font-size:1.2rem}h1{font-size:1.65rem!important}}</style>''',unsafe_allow_html=True)

ASSETS={'NIFTY 50':'^NSEI','India VIX':'^INDIAVIX','USD/INR':'INR=X','Brent Crude':'BZ=F','Dollar Index':'DX-Y.NYB','US 10Y Yield':'^TNX','Gold':'GC=F'}

def sf(v):
    try:
        x=float(v); return x if np.isfinite(x) else np.nan
    except: return np.nan

def close(raw,t,total):
    try:
        if total==1 and not isinstance(raw.columns,pd.MultiIndex): d=raw
        elif t in raw.columns.get_level_values(0): d=raw[t]
        else: d=raw.xs(t,axis=1,level=-1)
        return pd.to_numeric(d['Close'],errors='coerce').dropna()
    except: return pd.Series(dtype=float)

@st.cache_data(ttl=1200,show_spinner=False)
def macro_engine():
    ts=list(ASSETS.values())
    try: raw=yf.download(ts,period='2y',interval='1d',group_by='ticker',auto_adjust=False,threads=True,progress=False,timeout=25)
    except TypeError: raw=yf.download(ts,period='2y',interval='1d',group_by='ticker',auto_adjust=False,threads=True,progress=False)
    except: raw=pd.DataFrame()
    specs=[('NIFTY 50',1,2.2,8),('India VIX',-1,1.7,15),('USD/INR',-1,1.5,4),('Brent Crude',-1,1.5,12),('Dollar Index',-1,1,6),('US 10Y Yield',-1,1,8),('Gold',-1,.6,12)]
    rows=[]; num=den=0
    for n,d,w,s in specs:
        c=close(raw,ASSETS[n],len(ts))
        if len(c)<64: continue
        r1=(c.iloc[-1]/c.iloc[-22]-1)*100; r3=(c.iloc[-1]/c.iloc[-64]-1)*100
        sig=float(np.tanh(d*(.75*r3+.25*r1)/s)); num+=sig*w; den+=w
        rows.append({'Driver':n,'Latest':sf(c.iloc[-1]),'1M %':r1,'3M %':r3,'Contribution':sig*w,'As Of':pd.to_datetime(c.index[-1]).strftime('%Y-%m-%d')})
    score=float(np.clip(50+50*num/den,0,100)) if den else 50
    cov=int(round(100*den/sum(x[2] for x in specs)))
    df=pd.DataFrame(rows)
    def v(n,col='3M %'):
        z=df[df.Driver==n]; return sf(z.iloc[0][col]) if len(z) else np.nan
    nifty,crude,fx=v('NIFTY 50'),v('Brent Crude'),v('USD/INR'); vr=df[df.Driver=='India VIX']; vix=sf(vr.iloc[0].Latest) if len(vr) else np.nan
    infl=int(pd.notna(crude) and crude>10)+int(pd.notna(fx) and fx>3); stress=int(pd.notna(nifty) and nifty<-5)+int(pd.notna(vix) and vix>20)
    if score>=76 and (pd.isna(nifty) or nifty>0) and stress==0: regime='EARLY / RISK-ON'
    elif score>=68 and stress==0: regime='MID CYCLE / EXPANSION'
    elif infl>=1 and score>=38: regime='LATE CYCLE / INFLATION-SENSITIVE'
    elif score<38 or stress>=2: regime='RISK-OFF / CONTRACTION'
    else: regime='MID-TO-LATE / MIXED'
    newest=pd.to_datetime(df['As Of']).max() if len(df) else pd.NaT
    stale=(pd.Timestamp.now(tz=None).normalize()-newest).days>4 if pd.notna(newest) else True
    return score,regime,cov,df,stale,newest

def tech_score(r):
    s=str(r.get('Status','')); b=sf(r.get('Breakout %')); vr=sf(r.get('Volume Ratio'))
    x=95 if s=='Fresh Breakout' else 85 if '<=3M' in s else 62 if s=='Older Breakout' else np.clip(50+(b if pd.notna(b) else -5)*3,20,65)
    if pd.notna(vr): x+=np.clip(vr-1,-1,1)*5
    return float(np.clip(x,0,100))

def policy_score(industry):
    x=str(industry).lower()
    themes={'defence':90,'aerospace':90,'renewable':88,'power':84,'electrical':84,'capital goods':82,'infrastructure':82,'railway':85,'electronics':84,'semiconductor':88,'manufacturing':78,'telecom':72,'healthcare':70,'pharma':70,'bank':62,'financial':62,'auto':72,'chemical':62,'metal':62,'mining':60}
    hits=[v for k,v in themes.items() if k in x]
    return max(hits) if hits else 55

def sector_scores(scan,monthlies):
    out=[]
    for ind,g in scan.groupby('Industry'):
        r3=[]; r6=[]
        for sym in g.Symbol:
            m=monthlies.get(str(sym)+'.NS',pd.DataFrame())
            if m.empty or 'Close' not in m: continue
            c=pd.to_numeric(m.Close,errors='coerce').dropna()
            if len(c)>6: r3.append((c.iloc[-1]/c.iloc[-4]-1)*100); r6.append((c.iloc[-1]/c.iloc[-7]-1)*100)
        a=np.median(r3) if r3 else np.nan; b=np.median(r6) if r6 else np.nan
        recent=(g['Months Since Signal'].fillna(999)<=3).mean()*100
        mom=np.mean([50+50*np.tanh(a/10) if pd.notna(a) else 50,50+50*np.tanh(b/18) if pd.notna(b) else 50])
        out.append({'Industry':ind,'Sector Score':float(np.clip(.75*mom+.25*recent,0,100))})
    return pd.DataFrame(out)

@st.cache_data(ttl=3600,show_spinner=False)
def quality(ticker):
    t=yf.Ticker(ticker)
    try: info=t.info or {}
    except: info={}
    try: inc=t.financials.copy(); bs=t.balance_sheet.copy()
    except: inc=pd.DataFrame(); bs=pd.DataFrame()
    def row(df,names):
        if df.empty:return pd.Series(dtype=float)
        mp={str(i).lower():i for i in df.index}
        for n in names:
            if n.lower() in mp:
                s=pd.to_numeric(df.loc[mp[n.lower()]],errors='coerce').dropna(); s.index=pd.to_datetime(s.index,errors='coerce'); return s[s.index.notna()].sort_index()
        return pd.Series(dtype=float)
    rev=row(inc,['Total Revenue','Operating Revenue']); prof=row(inc,['Net Income','Net Income Common Stockholders']); ebit=row(inc,['EBIT','Operating Income']); assets=row(bs,['Total Assets']); cl=row(bs,['Current Liabilities','Total Current Liabilities']); eq=row(bs,['Stockholders Equity','Total Equity Gross Minority Interest']); debt=row(bs,['Total Debt'])
    def cagr(s):
        if len(s)<2 or s.iloc[0]<=0 or s.iloc[-1]<=0:return np.nan,0
        y=(s.index[-1]-s.index[0]).days/365.25; return (((s.iloc[-1]/s.iloc[0])**(1/y)-1)*100,y) if y>0 else (np.nan,0)
    sg,sy=cagr(rev); pg,py=cagr(prof); equity=sf(eq.iloc[-1]) if len(eq) else np.nan; td=sf(info.get('totalDebt'))
    if pd.isna(td) and len(debt):td=sf(debt.iloc[-1])
    de=td/equity if pd.notna(td) and pd.notna(equity) and equity else np.nan; common=assets.index.intersection(cl.index); cap=sf(assets.loc[common[-1]]-cl.loc[common[-1]]) if len(common) else np.nan; eb=sf(ebit.iloc[-1]) if len(ebit) else np.nan; roce=eb/cap*100 if pd.notna(eb) and pd.notna(cap) and cap else np.nan
    p=prof.dropna().sort_index(); incprofit=bool(len(p)>=3 and all(p.iloc[i]>=p.iloc[i-1] for i in range(1,len(p)))) if len(p)>=3 else None
    checks=[]
    for val,avail,fn in [(de,pd.notna(de),lambda z:z<.5),(roce,pd.notna(roce),lambda z:z>15),(incprofit,incprofit is not None,lambda z:z is True),(sg,pd.notna(sg) and sy>=5,lambda z:z>20),(pg,pd.notna(pg) and py>=5,lambda z:z>15)]:
        if avail:checks.append(int(fn(val)))
    q=100*sum(checks)/len(checks) if checks else np.nan; pe=sf(info.get('trailingPE'))
    valuation=np.nan if pd.isna(pe) or pe<=0 else float(np.clip(100-(pe-15)*2,15,95))
    return q,valuation,pe,len(checks)

def label(x):
    return 'STRONG OPPORTUNITY' if x>=82 else 'BUY-WATCH' if x>=72 else 'WATCH' if x>=58 else 'AVOID / LOW PRIORITY'

def risk_reason(r):
    risks=[]
    if r['Macro Score']<45:risks.append('weak macro')
    if r['Sector Score']<50:risks.append('weak sector')
    if r['Technical Score']<60:risks.append('weak/no fresh breakout')
    if pd.notna(r['Valuation Score']) and r['Valuation Score']<45:risks.append('expensive valuation')
    if pd.notna(r['Fundamental Score']) and r['Fundamental Score']<60:risks.append('fundamental gaps')
    return ', '.join(risks) if risks else 'no major model flag'

def why(r):
    parts=[]
    for name in ['Technical','Sector','Policy','Fundamental','Valuation']:
        v=r.get(name+' Score',np.nan)
        if pd.notna(v) and v>=70:parts.append(name.lower())
    return ' + '.join(parts[:3]) if parts else 'mixed factors'

st.title('🎯 Investment Decision Engine — 0 to 100')
st.caption('Transparent decision-support: Macro + Policy alignment + Sector strength + strict 26M ATH + Fundamentals + Valuation. Scores prioritise research; they are not buy/sell advice.')
ms,regime,mcov,drivers,stale,newest=macro_engine(); scan=st.session_state.get('pro_scan',pd.DataFrame()); monthlies=st.session_state.get('pro_monthlies',{}); stored=dict(st.session_state.get('decision_fund',{}))
a,b,c,d=st.columns(4); a.metric('Macro Score',f'{ms:.0f}/100'); text_metric(b,'Live Regime',regime); c.metric('Macro Coverage',f'{mcov}%'); d.metric('Scan Records',len(scan))
updated='N/A' if pd.isna(newest) else newest.strftime('%Y-%m-%d'); st.caption(f'Market data as-of: **{updated}** · page generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")} · macro cache TTL 20 min')
if stale:st.warning('⚠️ Macro market feed may be stale or incomplete. Treat scores as provisional and verify exchange/broker data.')
if mcov<80:st.warning(f'⚠️ Macro input coverage is only {mcov}%. Regime confidence is reduced.')
with st.expander('Why is this Macro Score / Regime?',False):
    if len(drivers): st.dataframe(drivers.sort_values('Contribution',ascending=False),use_container_width=True,hide_index=True)
    st.caption('Regime is recalculated from live/cached market proxies; it is not hard-coded and is not an official GDP-cycle classification.')
if scan.empty:
    st.warning('पहले Professional Research Lab → 26M ATH Scanner में scan चलाएँ। उसके बाद यह page stock-wise decision ranking बनाएगा।'); st.stop()
sec=sector_scores(scan,monthlies); smap=sec.set_index('Industry')['Sector Score'].to_dict() if len(sec) else {}; d=scan.copy(); d['Technical Score']=d.apply(tech_score,axis=1); d['Sector Score']=d.Industry.map(smap).fillna(50); d['Policy Score']=d.Industry.apply(policy_score); d['Macro Score']=ms; d['Fundamental Score']=d.Symbol.map(lambda x: stored.get(str(x),{}).get('q',np.nan)); d['Valuation Score']=d.Symbol.map(lambda x: stored.get(str(x),{}).get('v',np.nan)); d['PE']=d.Symbol.map(lambda x: stored.get(str(x),{}).get('pe',np.nan))
# fixed model weights; unavailable fundamental/valuation weights are redistributed across available components
weights={'Macro Score':.15,'Policy Score':.10,'Sector Score':.20,'Technical Score':.25,'Fundamental Score':.20,'Valuation Score':.10}
def final(r):
    avail=[(k,w) for k,w in weights.items() if pd.notna(r[k])]; return sum(r[k]*w for k,w in avail)/sum(w for _,w in avail)
d['Final Investment Score']=d.apply(final,axis=1); d['Decision']=d['Final Investment Score'].apply(label); d['Data Coverage %']=d.apply(lambda r:round(100*sum(w for k,w in weights.items() if pd.notna(r[k]))),axis=1); d['Why Ranked']=d.apply(why,axis=1); d['Risk Flags']=d.apply(risk_reason,axis=1); d=d.sort_values('Final Investment Score',ascending=False)
left,right=st.columns([2,1]); left.subheader('🏆 Final Opportunities'); n=right.selectbox('Deep fundamental + valuation scan',[5,10,20,30],index=1)
if right.button(f'Analyse Top {n}',type='primary',use_container_width=True):
    p=st.progress(0,text='Analysing fundamentals & valuation…')
    for i,sym in enumerate(d.head(n).Symbol.astype(str)):
        try:q,v,pe,ass=quality(sym+'.NS');stored[sym]={'q':q,'v':v,'pe':pe,'assessed':ass}
        except:stored[sym]={'q':np.nan,'v':np.nan,'pe':np.nan,'assessed':0}
        p.progress((i+1)/n,text=f'{i+1}/{n}: {sym}')
    p.empty();st.session_state['decision_fund']=stored;st.rerun()
show=['Symbol','Company','Industry','Decision','Final Investment Score','Macro Score','Policy Score','Sector Score','Technical Score','Fundamental Score','Valuation Score','PE','Data Coverage %','Why Ranked','Risk Flags']
st.dataframe(d[show].head(50),use_container_width=True,hide_index=True,column_config={'Final Investment Score':st.column_config.ProgressColumn(min_value=0,max_value=100,format='%.0f'),'Macro Score':st.column_config.ProgressColumn(min_value=0,max_value=100,format='%.0f'),'Policy Score':st.column_config.ProgressColumn(min_value=0,max_value=100,format='%.0f'),'Sector Score':st.column_config.ProgressColumn(min_value=0,max_value=100,format='%.0f'),'Technical Score':st.column_config.ProgressColumn(min_value=0,max_value=100,format='%.0f'),'Fundamental Score':st.column_config.ProgressColumn(min_value=0,max_value=100,format='%.0f'),'Valuation Score':st.column_config.ProgressColumn(min_value=0,max_value=100,format='%.0f'),'Data Coverage %':st.column_config.ProgressColumn(min_value=0,max_value=100,format='%d%%')})
x1,x2,x3,x4=st.columns(4);x1.metric('Strong Opportunity',int((d.Decision=='STRONG OPPORTUNITY').sum()));x2.metric('Buy-Watch',int((d.Decision=='BUY-WATCH').sum()));x3.metric('Watch',int((d.Decision=='WATCH').sum()));x4.metric('Full-data Names',int((d['Data Coverage %']==100).sum()))
with st.expander('⚙️ Score methodology & limitations'):
    st.markdown('**Weights:** Macro 15%, Policy 10%, Sector 20%, 26M ATH Technical 25%, Fundamentals 20%, Valuation 10%. Missing fundamental/valuation components are excluded and remaining weights are normalized. Policy score is a transparent industry-theme heuristic, not a live government-policy database. Valuation currently uses trailing P/E when available; sector-relative valuation should be verified independently.')
    st.markdown('**Decision bands:** 82+ Strong Opportunity · 72–81 Buy-Watch · 58–71 Watch · below 58 Avoid/Low Priority.')
st.info('Risk control: verify NSE/BSE prices, filings, promoter/shareholding, corporate actions and valuation before acting. A high score is a research priority, not a guaranteed return or recommendation.')