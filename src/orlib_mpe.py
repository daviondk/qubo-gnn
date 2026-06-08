"""Chang et al. (2000) Mean Percentage Error (MPE) metric for OR-Library cardinality frontiers,
so our methods can be compared to the MPE-family papers (Chang GA/SA/TS, Deng IPSO, ARO).

MPE = mean over E frontier points of min(horizontal%, vertical%) deviation from the unconstrained
efficient frontier, in (standard-deviation, return) space (Chang uses std, not variance).
"""
from __future__ import annotations

import sys, os, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from portfolio_data import download_orlib, load_orlib
from baselines import scip_cardinality, convex_reweight
from orlib_metrics import unconstrained_frontier
from qubo_portfolio import selection_qubo, decode_selection

EPS, DELTA = 0.01, 1.0


def efficient_hull(std, ret):
    """Upper-left efficient frontier as monotonic arrays sorted by std (running-max return)."""
    o = np.argsort(std); s = std[o]; r = ret[o]
    keep_s, keep_r, best = [], [], -np.inf
    for si, ri in zip(s, r):
        if ri > best + 1e-12:
            keep_s.append(si); keep_r.append(ri); best = ri
    return np.array(keep_s), np.array(keep_r)


def chang_mpe(card_std, card_ret, uef_std, uef_ret):
    s, r = efficient_hull(uef_std, uef_ret)
    if len(s) < 2:
        return float("nan")
    errs = []
    rmin, rmax = r.min(), r.max(); smin, smax = s.min(), s.max()
    for xs, yr in zip(card_std, card_ret):
        e = []
        if rmin <= yr <= rmax:                       # horizontal: same return -> compare std
            xstar = np.interp(yr, r, s)
            if xstar > 1e-12:
                e.append(abs(100 * (xs - xstar) / xstar))
        if smin <= xs <= smax:                       # vertical: same std -> compare return
            ystar = np.interp(xs, s, r)
            if abs(ystar) > 1e-12:
                e.append(abs(100 * (yr - ystar) / ystar))
        if e:
            errs.append(min(e))
    return float(np.mean(errs)) if errs else float("nan")


def card_frontier_exact(mu, Sigma, K, lams):
    pts = []
    for lam in lams:
        r = scip_cardinality(mu, Sigma, K, risk_aversion=float(lam), return_weight=float(1 - lam),
                             eps=EPS, delta=DELTA, time_limit=30)
        w = r["weights"]; pts.append((np.sqrt(max(w @ Sigma @ w, 0)), float(mu @ w)))
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


PUBLISHED_MPE = {  # Chang 2000 / Deng 2012 / ARO 2021
    "port1": {"GA": 1.0974, "SA": 1.0957, "TS": 1.1217, "IPSO": 1.0953, "ARO": 1.4181},
    "port2": {"GA": 2.5424, "SA": 2.9297, "TS": 3.3049, "IPSO": 2.5417, "ARO": 1.3190},
    "port3": {"GA": 1.1076, "SA": 1.4623, "TS": 1.1217, "IPSO": 1.0628, "ARO": 0.8151},
    "port4": {"GA": 1.9328, "SA": 3.0696, "TS": 3.3092, "IPSO": 1.6890, "ARO": 1.4468},
    "port5": {"GA": 0.7961, "SA": 0.6732, "TS": 0.8975, "IPSO": 0.6870, "ARO": 0.6179},
}


def main():
    names = sys.argv[1:] if len(sys.argv) > 1 else ["port1", "port2", "port3", "port4", "port5"]
    K = 10
    lams = np.linspace(0.0, 1.0, 50)
    paths = download_orlib("data/orlib")
    out = {}
    for name in names:
        mu, Sig, _ = load_orlib(paths[name])
        uv, ur = unconstrained_frontier(mu, Sig, n_points=2000)
        ustd = np.sqrt(np.clip(uv, 0, None))
        cs, cr = card_frontier_exact(mu, Sig, K, lams)
        mpe = chang_mpe(cs, cr, ustd, ur)
        out[name] = mpe
        pub = PUBLISHED_MPE[name]
        print(f"{name}: OURS(exact-frontier MPE)={mpe:.4f}%  | published: "
              + " ".join(f"{k}={v}" for k, v in pub.items()), flush=True)
    os.makedirs("results/orlib_med", exist_ok=True)
    json.dump(out, open("results/orlib_med/mpe_exact.json", "w"), indent=2)


if __name__ == "__main__":
    main()
