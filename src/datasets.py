"""Extra public datasets for portfolio experiments (beyond OR-Library + S&P100/DOW30):
French 49-Industry (Ken French data library, public), NASDAQ-100, crypto, Russell-1000 sample.
Each loader returns a daily-returns DataFrame (fraction). Cached under data/.
"""
from __future__ import annotations

import os, io, zipfile, urllib.request
import numpy as np
import pandas as pd

FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_daily_CSV.zip"

NASDAQ100 = ["AAPL","MSFT","AMZN","NVDA","GOOGL","META","AVGO","PEP","COST","CSCO","TMUS","ADBE","TXN",
    "AMD","QCOM","AMGN","INTC","HON","INTU","AMAT","BKNG","SBUX","GILD","ADI","MDLZ","REGN","VRTX","LRCX",
    "PYPL","MU","PANW","SNPS","CDNS","KLAC","MAR","ORLY","CSX","ASML","ABNB","CHTR","FTNT","MNST","ADP",
    "NXPI","PCAR","KDP","MRVL","AEP","ODFL","ROST","KHC","IDXX","CTAS","EA","EXC","DXCM","BIIB","CPRT",
    "FAST","XEL","WBD","CCEP","ON","GEHC","CSGP","DLTR","ANSS","TTD","VRSK","WDAY","BKR","CTSH","TEAM"]

CRYPTO = ["BTC-USD","ETH-USD","BNB-USD","XRP-USD","ADA-USD","SOL-USD","DOGE-USD","DOT-USD","LTC-USD",
    "BCH-USD","LINK-USD","XLM-USD","ATOM-USD","ETC-USD","XMR-USD","ALGO-USD","VET-USD","FIL-USD",
    "EOS-USD","AAVE-USD","XTZ-USD","THETA-USD","AVAX-USD","TRX-USD"]


def load_french49(cache_dir="data"):
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, "french49_daily.pkl")
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    raw = urllib.request.urlopen(FRENCH_URL, timeout=60).read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
    text = zf.read(name).decode("latin-1").splitlines()
    rows = []
    for ln in text:
        parts = ln.split(",")
        if len(parts) >= 50 and parts[0].strip().isdigit() and len(parts[0].strip()) == 8:
            rows.append(parts[:50])
    cols = ["date"] + [f"Ind{i}" for i in range(49)]
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").astype(float) / 100.0
    df = df.replace(-0.9999, np.nan).replace(-99.99 / 100, np.nan).dropna()
    df.to_pickle(cache)
    return df


def load_yf_returns(tickers, name, start="2015-01-01", end="2024-12-31", cache_dir="data"):
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"{name}_{start}_{end}.pkl")
    if os.path.exists(cache):
        px = pd.read_pickle(cache)
    else:
        import yfinance as yf
        px = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
        px = px.dropna(axis=1, how="any").dropna()
        px.to_pickle(cache)
    return px.pct_change().dropna()


def sp500_tickers(cache_dir="data"):
    """Current S&P 500 constituents from Wikipedia (cached)."""
    import pandas as pd
    cache = os.path.join(cache_dir, "sp500_tickers.txt")
    if os.path.exists(cache):
        return open(cache).read().split()
    import io
    req = urllib.request.Request("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                                 headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
    tabs = pd.read_html(io.StringIO(html))
    tk = [str(t).replace(".", "-") for t in tabs[0]["Symbol"].tolist()]
    open(cache, "w").write(" ".join(tk))
    return tk


def get_returns(name):
    if name == "french49":
        return load_french49()
    if name == "nasdaq100":
        return load_yf_returns(NASDAQ100, "nasdaq100")
    if name == "crypto":
        return load_yf_returns(CRYPTO, "crypto", start="2019-01-01")
    if name == "sp500":
        return load_yf_returns(sp500_tickers(), "sp500", start="2015-01-01")
    raise ValueError(name)


if __name__ == "__main__":
    for nm in ["french49", "nasdaq100", "crypto"]:
        try:
            r = get_returns(nm)
            print(f"{nm}: {r.shape[1]} assets x {r.shape[0]} days  {r.index[0].date()}->{r.index[-1].date()}")
        except Exception as e:
            print(f"{nm}: FAILED {str(e)[:80]}")
