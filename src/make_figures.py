"""Generate all paper figures from the experiment results into results/figures/*.png.
Numbers sourced from the docs (docs/08,13,14,15,16,17) — the canonical verified values.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/figures"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})
INST = ["port1", "port2", "port3", "port4", "port5"]


def fig_med():
    med = {
        "GNN+refine (ours)": [0.0001, 0.0002, 0.0000, 0.0001, 0.0000],
        "QRF-GNN orig (ours)": [0.0001, 0.0004, 0.0003, 0.0004, 0.0004],
        "IPSO-SA (2011)": [0.0001, 0.0001, 0.0000, 0.0001, 0.0000],
        "Firefly (2014)": [0.0003, 0.0009, 0.0004, 0.0003, 0.0000],
        "GA (Cura 2009)": [0.0040, 0.0076, 0.0020, 0.0041, 0.0093],
        "PSO (Cura 2009)": [0.0049, 0.0090, 0.0022, 0.0052, 0.0024],
    }
    x = np.arange(5); w = 0.13
    plt.figure(figsize=(10, 5))
    for i, (k, v) in enumerate(med.items()):
        plt.bar(x + (i - 2.5) * w, np.array(v) + 1e-5, w, label=k)
    plt.yscale("log"); plt.xticks(x, INST); plt.ylabel("MED (log, lower=better)")
    plt.title("OR-Library cardinality MED: ours vs published (Cura metric, K=10)")
    plt.legend(fontsize=8, ncol=2); plt.grid(True, axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(f"{OUT}/fig_med_comparison.png"); plt.close()


def fig_scaling():
    N = [500, 1000, 1500, 2000]
    SA = [58.7, 58.5, 64.5, 62.5]; Tabu = [0.001, 0.001, 0.001, 0.001]; GNN = [0.001, 0.002, 0.001, 0.001]
    plt.figure(figsize=(7, 5))
    plt.plot(N, SA, "s-", label="SA", color="blue")
    plt.plot(N, Tabu, "^-", label="Tabu", color="green")
    plt.plot(N, GNN, "*-", label="GNN (ours)", color="red", markersize=12)
    plt.axhline(1e-3, ls=":", color="gray")
    plt.text(500, 200, "SCIP-global: fails (timeout) at all N", color="purple", fontsize=9)
    plt.yscale("symlog", linthresh=1e-3); plt.xlabel("N (assets, dense QUBO)")
    plt.ylabel("optimality gap % (log)"); plt.title("Scaling: QUBO solver quality vs N")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(f"{OUT}/fig_scaling.png"); plt.close()


def fig_amortized():
    labels = ["S&P100\n(in-dist)", "NASDAQ100\n(OOD)", "French49\n(OOD,diffN)"]
    gap = [0.87, 0.49, 2.11]; med = [0.67, 0.68, 0.71]; speed = [844, 1107, 1121]
    fig, ax1 = plt.subplots(figsize=(7.5, 5)); x = np.arange(3)
    ax1.bar(x - 0.2, gap, 0.4, label="mean gap %", color="salmon")
    ax1.bar(x + 0.2, med, 0.4, label="median gap %", color="lightgreen")
    ax1.set_ylabel("gap vs per-instance tabu (%)"); ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax2 = ax1.twinx(); ax2.plot(x, speed, "ko--", label="speedup×"); ax2.set_ylabel("speedup × (per-instance)")
    for xi, s in zip(x, speed):
        ax2.annotate(f"{s}×", (xi, s), textcoords="offset points", xytext=(0, 6), fontsize=9)
    ax1.set_title("Amortized GNN: train on S&P100, deploy OOD (≈tabu quality, ~1000× faster)")
    ax1.legend(loc="upper left", fontsize=8); ax1.grid(True, axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(f"{OUT}/fig_amortized.png"); plt.close()


def fig_cvar():
    data = {
        "french49\n(N49,400sc)": {"exact": 0.01629, "EqualW": 0.02683, "Tabu+LP": 0.01764, "GNN+LP": 0.01764},
        "nasdaq100\n(N66,400sc)": {"exact": 0.01100, "EqualW": 0.01993, "Tabu+LP": 0.01242, "GNN+LP": 0.01238},
        "synthetic\n(N200,3000sc)": {"exact*": 0.00543, "EqualW": 0.01411, "Tabu+LP": 0.00516, "GNN+LP": 0.00538},
    }
    methods = ["exact", "EqualW", "Tabu+LP", "GNN+LP"]; colors = ["black", "orange", "green", "red"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, (inst, d) in zip(axs, data.items()):
        ks = list(d.keys()); vs = list(d.values())
        ax.bar(range(len(ks)), vs, color=[colors[min(i, 3)] for i in range(len(ks))])
        ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks, fontsize=8, rotation=20)
        ax.set_title(inst, fontsize=9); ax.set_ylabel("CVaR(95%) (lower=better)")
        ax.grid(True, axis="y", alpha=0.3)
    axs[2].annotate("exact MILP\nTIMES OUT\n→ hybrid wins", (0, 0.00543), color="purple", fontsize=8,
                    xytext=(0.5, 0.010), arrowprops=dict(arrowstyle="->", color="purple"))
    fig.suptitle("CVaR(95%) cardinality: hybrid (select+CVaR-LP) vs exact MILP — wins at scale")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_cvar.png"); plt.close()


def fig_backtest():
    strat = ["EqualWeight", "MinVar", "MaxSharpe", "Markowitz", "SA-card", "GNN-card"]
    sharpe = [0.813, 0.700, 0.750, 0.823, 0.805, 0.823]
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(strat, sharpe, color=["gray", "skyblue", "khaki", "lightgreen", "salmon", "red"])
    plt.ylabel("Sharpe (OOS 2006-2024)"); plt.title("S&P100 backtest (K=15): GNN-card = full Markowitz with 15/71 assets")
    plt.xticks(rotation=20); plt.ylim(0.6, 0.86); plt.grid(True, axis="y", alpha=0.3)
    for b, s in zip(bars, sharpe):
        plt.annotate(f"{s:.3f}", (b.get_x() + b.get_width() / 2, s), ha="center", va="bottom", fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_backtest.png"); plt.close()


def fig_datasets():
    ds = ["French49", "NASDAQ100", "crypto"]
    gnn = [0.014, 0.000, 13.22]; tabu = [0.014, 0.000, 13.22]; sa = [3.81, 3.01, 58.0]
    x = np.arange(3); w = 0.25
    plt.figure(figsize=(7.5, 4.5))
    plt.bar(x - w, gnn, w, label="GNN (ours)", color="red")
    plt.bar(x, tabu, w, label="Tabu", color="green")
    plt.bar(x + w, sa, w, label="SA", color="blue")
    plt.yscale("symlog", linthresh=0.01); plt.xticks(x, ds); plt.ylabel("regret % vs SCIP-exact (log)")
    plt.title("Cardinality on extra datasets (French49 = Lozano-2026 dataset)")
    plt.legend(); plt.grid(True, axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(f"{OUT}/fig_datasets.png"); plt.close()


if __name__ == "__main__":
    fig_med(); fig_scaling(); fig_amortized(); fig_cvar(); fig_backtest(); fig_datasets()
    print("figures written to", OUT)
    for f in sorted(os.listdir(OUT)):
        print(" ", f)
