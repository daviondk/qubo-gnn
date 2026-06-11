"""E108: Max-3-Cut on Gset vs ROS (arXiv:2412.05146, Table 7). SAME data (standard public Gset), SAME metric
(cut value, higher=better). Max-3-Cut (k=3 partition) is a DISTINCT problem from k=2 MaxCut (QIGNN).
Our unsupervised relaxation (softmax k=3 + cut energy) + node-move local search. Run in .venv."""
import sys, numpy as np, torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"
K = 3
REF = {  # ROS Table 7 (Max-3-Cut): MD, Genetic, BQP, ANYCSP, MOH(best), ROS
 "G14": dict(MD=3844, Genetic=3679, BQP=3900, ANYCSP=3973, MOH=4012, ROS=3892, V=800),
 "G15": dict(MD=3815, Genetic=3625, BQP=3885, ANYCSP=3975, MOH=3984, ROS=3838, V=800),
 "G22": dict(MD=16402, BQP=16599, ANYCSP=17098, MOH=17167, ROS=16601, V=2000),
}


def load_gset(path):
    rows = [l.split() for l in open(path) if l.strip()]
    if len(rows[0]) == 2:  # header N M
        N = int(rows[0][0]); E = rows[1:]
    else:
        E = rows; N = max(max(int(a), int(b)) for a, b, *_ in E)
    ei = np.array([[int(a) - 1, int(b) - 1] for a, b, *_ in E], dtype=np.int64)
    w = np.array([float(r[2]) if len(r) > 2 else 1.0 for r in E], dtype=np.float64)
    return N, ei, w


def cut_value(cls, ei, w):
    return float(w[cls[ei[:, 0]] != cls[ei[:, 1]]].sum())


def node_move_ls(cls, N, ei, w):
    # vectorized greedy: repeatedly move each node to class with least same-affinity
    u = np.concatenate([ei[:, 0], ei[:, 1]]); v = np.concatenate([ei[:, 1], ei[:, 0]]); ww = np.concatenate([w, w])
    improved = True; passes = 0
    while improved and passes < 30:
        improved = False; passes += 1
        clsw = np.zeros((N, K))
        np.add.at(clsw, (u, cls[v]), ww)  # clsw[node,c] = weight to neighbors in class c
        newc = clsw.argmin(1)
        moved = newc != cls
        if moved.any():
            # apply moves where it strictly helps (gain = clsw[old]-clsw[new] > 0)
            gain = clsw[np.arange(N), cls] - clsw[np.arange(N), newc]
            do = moved & (gain > 1e-9)
            if do.any(): cls[do] = newc[do]; improved = True
    return cls


def solve(N, ei, w, restarts=6, epochs=1500):
    E0 = torch.tensor(ei[:, 0], device=DEV); E1 = torch.tensor(ei[:, 1], device=DEV)
    wt = torch.tensor(w, dtype=torch.float32, device=DEV)
    best = -1; bestc = None
    for r in range(restarts):
        torch.manual_seed(r); logits = torch.randn(N, K, device=DEV, requires_grad=True)
        opt = torch.optim.Adam([logits], lr=0.05)
        for ep in range(epochs):
            T = max(0.2, 1.0 - ep / epochs)
            p = torch.softmax(logits / T, 1)
            agree = (p[E0] * p[E1]).sum(1)  # prob same class per edge
            loss = (wt * agree).sum()  # minimize same-class -> maximize cut
            opt.zero_grad(); loss.backward(); opt.step()
        cls = torch.softmax(logits, 1).argmax(1).cpu().numpy().astype(np.int64)
        cls = node_move_ls(cls, N, ei, w)
        c = cut_value(cls, ei, w)
        if c > best: best = c; bestc = cls.copy()
    return int(round(best))


def main():
    print("=== Max-3-Cut on Gset vs ROS Table 7 (cut value, higher=better) ===", flush=True)
    for g in ["G14", "G15", "G22"]:
        N, ei, w = load_gset(f"Gset/{g}.txt"); ref = REF[g]
        ours = solve(N, ei, w)
        learned_best = max(ref["ANYCSP"], ref["ROS"])  # best LEARNED
        overall_best = ref["MOH"]
        vL = "BEAT learned" if ours > learned_best else ("~learned" if ours >= ref["ROS"] else "behind")
        vO = "BEAT MOH(overall)" if ours > overall_best else ""
        print(f"{g}: OURS {ours} | ROS {ref['ROS']} ANYCSP {ref['ANYCSP']} MOH {ref['MOH']} -> {vL} {vO}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
