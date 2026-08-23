"""Granular sector buckets for the NIFTY 500 26M ATH scanner.

The official NIFTY 500 constituent file uses broad Industry labels.  For a
research scanner it is useful to split several investable groups that have
separate NIFTY sectoral/thematic indices.  This module rewrites only the
scanner-facing Industry label; price/breakout logic is untouched.
"""
import pandas as pd

PHARMA = {
    "ABBOTINDIA","AJANTPHARM","ALKEM","APLLTD","ASTRAZEN","AUROPHARMA","BIOCON",
    "CAPLIPOINT","CIPLA","CONCORDBIO","DIVISLAB","DRREDDY","ERIS","FDC","GLAND",
    "GLAXO","GLENMARK","GRANULES","IPCALAB","JBCHEPHARM","LAURUSLABS","LUPIN",
    "MANKIND","NATCOPHARM","NEULANDLAB","PFIZER","PIRAMALPHARMA","SANOFI",
    "SUNPHARMA","SYNGENE","TORNTPHARM","ZYDUSLIFE"
}
HOSPITALS = {
    "APOLLOHOSP","ASTERDM","FORTIS","KIMS","LALPATHLAB","MAXHEALTH","MEDANTA",
    "METROPOLIS","NH","THYROCARE","VIJAYA"
}
DEFENCE = {
    "HAL","BEL","BDL","BEML","MAZDOCK","COCHINSHIP","GRSE","DATAPATTNS","PARAS",
    "SOLARINDS","ASTRAMICRO","BHARATFORG","MIDHANI","MTARTECH","DCXINDIA","ZENTEC",
    "IDEAFORGE","DYNAMATECH"
}
PRIVATE_BANKS = {
    "HDFCBANK","ICICIBANK","AXISBANK","KOTAKBANK","INDUSINDBK","FEDERALBNK","IDFCFIRSTB",
    "BANDHANBNK","YESBANK","RBLBANK","CSBBANK","DCBBANK","CUB","KARURVYSYA","SOUTHBANK",
    "AUBANK"
}
PSU_BANKS = {
    "SBIN","BANKBARODA","PNB","CANBK","UNIONBANK","INDIANB","BANKINDIA","MAHABANK",
    "CENTRALBK","IOB","UCOBANK","PSB"
}
INSURANCE = {
    "HDFCLIFE","SBILIFE","ICICIPRULI","ICICIGI","STARHEALTH","GICRE","NIACL","LICI"
}
NBFC = {
    "BAJFINANCE","SHRIRAMFIN","CHOLAFIN","M&MFIN","LTF","PFC","RECLTD","MUTHOOTFIN",
    "MANAPPURAM","IIFL","POONAWALLA","ABCAPITAL","CREDITACC","SBFC","360ONE","MOTILALOFS"
}
HOUSING_FINANCE = {
    "BAJAJHFL","AAVAS","APTUS","CANFINHOME","HOMEFIRST","HUDCO","LICHSGFIN","PNBHOUSING",
    "SAMMAANCAP","AADHARHFC"
}
CEMENT = {
    "ULTRACEMCO","GRASIM","SHREECEM","AMBUJACEM","JKCEMENT","DALBHARAT","RAMCOCEM","ACC",
    "JKLAKSHMI","NUVOCO","BIRLACORPN","HEIDELBERG","PRSMJOHNSN","ORIENTCEM","STARCEMENT","SAGCEM"
}
RETAIL = {
    "TRENT","DMART","NYKAA","VMM","SHOPERSTOP","ABFRL","ABLBL","VMART","MANYAVAR"
}


def apply_scanner_sector_classification(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with more useful scanner-level sector labels."""
    if not isinstance(df, pd.DataFrame) or "Symbol" not in df.columns:
        return df
    out = df.copy()
    out["Symbol"] = out["Symbol"].astype(str).str.strip()
    if "Company Name" not in out.columns:
        out["Company Name"] = out["Symbol"]

    # Specific groups first; each stock receives one primary scanner bucket.
    rules = [
        ("Pharmaceuticals", PHARMA),
        ("Hospitals & Diagnostics", HOSPITALS),
        ("Defence & Aerospace", DEFENCE),
        ("Private Banks", PRIVATE_BANKS),
        ("PSU Banks", PSU_BANKS),
        ("Insurance", INSURANCE),
        ("Housing Finance", HOUSING_FINANCE),
        ("NBFC", NBFC),
        ("Cement", CEMENT),
        ("Retail", RETAIL),
    ]
    for label, symbols in rules:
        out.loc[out["Symbol"].isin(symbols), "Industry"] = label

    # Catch remaining banks without disturbing the dedicated private/PSU sets.
    names = out["Company Name"].astype(str)
    bank_mask = names.str.contains(r"\bBank\b", case=False, regex=True, na=False)
    dedicated = out["Symbol"].isin(PRIVATE_BANKS | PSU_BANKS)
    out.loc[bank_mask & ~dedicated, "Industry"] = "Banking"
    return out
