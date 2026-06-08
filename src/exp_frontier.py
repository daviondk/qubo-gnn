"""Trace the cardinality-constrained efficient frontier (standard OR-Library evaluation).

Sweep the risk/return tradeoff lambda in (0,1): objective = lambda*w'Sigma w - (1-lambda)*mu'w.
For each lambda, solve with MIQP (if license allows), GNN, SA, Greedy; record (vol, return) and the
objective. Compute each method's mean optimality gap vs the reference (MIQP if available, else
best-found) and plot the frontiers against the unconstrained convex frontier.
"""
from __future__ import annotations

import sys, os, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from portfolio_data import download_orlib, load_orlib, ORLIB_FILES
from exp_cardinality import run_point


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "port2"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    n_pts = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    paths = download_orlib("data/orlib")
    mu, Sigma, _ = load_orlib(paths[name])
    lambdas = np.linspace(0.1, 0.95, n_pts)

    methods = ["MIQP(exact)", "GNN", "SA", "Greedy"]
    front = {m: {"ret": [], "vol": [], "gap": []} for m in methods}
    exact_ok = True
    for lam in lambdas:
        res = run_point(mu, Sigma, K, ra=float(lam), rw=float(1 - lam), gnn_epochs=1200)
        if not res["_ref"]["exact"]:
            exact_ok = False
        for m in methods:
            d = res.get(m, {})
            front[m]["ret"].append(d.get("return", np.nan))
            front[m]["vol"].append(d.get("vol", np.nan))
            front[m]["gap"].append(d.get("gap", np.nan))
        print(f"lam={lam:.2f}  " + "  ".join(
            f"{m}:gap={np.array(front[m]['gap'])[-1]*100:6.2f}%" for m in ["GNN", "SA", "Greedy"]))

    # summary: mean optimality gap per method (exclude reference itself)
    print(f"\n=== {name} N={len(mu)} K={K} | reference={'exact MIQP' if exact_ok else 'best-found'} ===")
    summary = {}
    for m in methods:
        g = np.array(front[m]["gap"], float)
        g = g[np.isfinite(g)]
        summary[m] = {"mean_gap_pct": float(np.mean(g) * 100) if len(g) else float("nan"),
                      "max_gap_pct": float(np.max(g) * 100) if len(g) else float("nan")}
        print(f"  {m:<12} mean_gap={summary[m]['mean_gap_pct']:.3f}%  max_gap={summary[m]['max_gap_pct']:.3f}%")

    # plot
    os.makedirs("results/frontier", exist_ok=True)
    plt.figure(figsize=(8, 6))
    colors = {"MIQP(exact)": "k", "GNN": "red", "SA": "blue", "Greedy": "green"}
    markers = {"MIQP(exact)": "o", "GNN": "*", "SA": "s", "Greedy": "^"}
    for m in methods:
        v = np.array(front[m]["vol"]); r = np.array(front[m]["ret"])
        ok = np.isfinite(v) & np.isfinite(r)
        if ok.any():
            plt.scatter(v[ok], r[ok], c=colors[m], marker=markers[m], s=70 if m != "GNN" else 120,
                        label=m, edgecolors="black", alpha=0.8, zorder=5 if m == "GNN" else 3)
    plt.xlabel("Volatility"); plt.ylabel("Expected return")
    plt.title(f"Cardinality-constrained frontier ({name}, N={len(mu)}, K={K})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    out_png = f"results/frontier/{name}_K{K}.png"
    plt.savefig(out_png, dpi=120)
    with open(f"results/frontier/{name}_K{K}.json", "w") as f:
        json.dump({"lambdas": lambdas.tolist(), "front": front, "summary": summary,
                   "exact": exact_ok, "N": len(mu), "K": K}, f, indent=2, default=str)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
