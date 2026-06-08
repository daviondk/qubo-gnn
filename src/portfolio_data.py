"""Portfolio datasets: Beasley OR-Library (cardinality benchmark) + yfinance S&P panel.

OR-Library port1..port5 file format (Chang, Meade, Beasley, Sharaiha 2000):
  line 1: N (number of assets)
  next N lines: <mean_return> <std_dev>           (per asset)
  remaining lines: i j corr_ij  (1-indexed, upper triangle incl. diagonal)
  covariance_ij = corr_ij * std_i * std_j
"""
from __future__ import annotations

import os
import urllib.request

import numpy as np

ORLIB_URL = "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/{}"
ORLIB_FILES = {  # name -> (filename, index, n_assets)
    "port1": ("port1.txt", "Hang Seng", 31),
    "port2": ("port2.txt", "DAX 100", 85),
    "port3": ("port3.txt", "FTSE 100", 89),
    "port4": ("port4.txt", "S&P 100", 98),
    "port5": ("port5.txt", "Nikkei 225", 225),
}


def download_orlib(out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, (fname, _, _) in ORLIB_FILES.items():
        dest = os.path.join(out_dir, fname)
        if not os.path.exists(dest) or os.path.getsize(dest) < 100:
            url = ORLIB_URL.format(fname)
            urllib.request.urlretrieve(url, dest)
        paths[name] = dest
    return paths


def load_orlib(path: str):
    """Return (mu [N], Sigma [N,N], names[list])."""
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it))
    mu = np.zeros(n)
    sd = np.zeros(n)
    for i in range(n):
        mu[i] = float(next(it))
        sd[i] = float(next(it))
    corr = np.eye(n)
    # remaining triples
    rest = list(it)
    for k in range(0, len(rest), 3):
        i = int(rest[k]) - 1
        j = int(rest[k + 1]) - 1
        c = float(rest[k + 2])
        corr[i, j] = c
        corr[j, i] = c
    Sigma = corr * np.outer(sd, sd)
    Sigma = 0.5 * (Sigma + Sigma.T)
    names = [f"A{i}" for i in range(n)]
    return mu, Sigma, names


def load_yfinance_panel(tickers, period="5y", interval="1d", cache_dir="data", annualize=True):
    """Download/caches a price panel; returns (mu, Sigma, tickers_kept). Network required."""
    import pandas as pd
    os.makedirs(cache_dir, exist_ok=True)
    key = f"prices_{len(tickers)}_{period}_{interval}.parquet"
    cache = os.path.join(cache_dir, key)
    if os.path.exists(cache):
        prices = pd.read_parquet(cache)
    else:
        import yfinance as yf
        prices = yf.download(tickers, period=period, interval=interval, auto_adjust=True)["Close"]
        prices = prices.dropna(axis=1, how="any").dropna()
        prices.to_parquet(cache)
    rets = prices.pct_change().dropna()
    mu = rets.mean().values
    Sigma = rets.cov().values
    if annualize:
        factor = 252 if interval == "1d" else 1
        mu = mu * factor
        Sigma = Sigma * factor
    Sigma = 0.5 * (Sigma + Sigma.T)
    return mu, Sigma, list(rets.columns)
