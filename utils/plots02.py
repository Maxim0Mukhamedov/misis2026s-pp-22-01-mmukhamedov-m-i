import csv
from collections import defaultdict
import matplotlib.pyplot as plt

def load_rows(path="results.csv"):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            row["N"] = int(row["N"])
            row["S"] = int(row["S"])
            row["M"] = int(row["M"])
            row["t_sec"] = float(row["t_sec"])
            row["gflops"] = float(row["gflops"])
            row["t_transpose_sec"] = float(row["t_transpose_sec"])
            row["gflops_no_transpose"] = float(row["gflops_no_transpose"])
            rows.append(row)
    return rows

rows = load_rows()

def plot_transpose_vs_classic():
    # пункт 6 [1]
    Ns = sorted({r["N"] for r in rows})
    classic = {r["N"]: r["gflops"] for r in rows if r["alg"]=="classic_ijk"}
    tr_with = {r["N"]: r["gflops"] for r in rows if r["alg"]=="transposeB_ijk"}
    tr_wo   = {r["N"]: r["gflops_no_transpose"] for r in rows if r["alg"]=="transposeB_ijk"}

    x = [N for N in Ns if N in classic and N in tr_with and N in tr_wo]
    plt.figure()
    plt.plot(x, [classic[N] for N in x], marker="o", label="classic_ijk")
    plt.plot(x, [tr_with[N] for N in x], marker="o", label="transposeB (with transpose time)")
    plt.plot(x, [tr_wo[N] for N in x], marker="o", label="transposeB (mul only)")
    plt.xlabel("N"); plt.ylabel("GFLOP/s")
    plt.title("Transpose(B) vs classic")
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig("p6_transpose_vs_classic.png", dpi=200)

def plot_buffer_M(N0=1024):
    # пункт 7 [1]
    pts = [(r["M"], r["gflops"]) for r in rows if r["alg"]=="buffer_colB" and r["N"]==N0]
    pts.sort()
    if not pts: return
    M = [p[0] for p in pts]; G = [p[1] for p in pts]
    bestM = pts[max(range(len(pts)), key=lambda i: pts[i][1])][0]

    plt.figure()
    plt.plot(M, G, marker="o")
    plt.xlabel("M"); plt.ylabel("GFLOP/s")
    plt.title(f"buffer_colB: P_real(M) at N={N0}, best M*={bestM}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"p7_buffer_P_vs_M_N{N0}.png", dpi=200)

def plot_blocked_S_bestM(N0=1024):
    # пункт 8: P(S) при лучшем M для каждого S [1]
    byS = defaultdict(list)
    for r in rows:
        if r["alg"]=="blocked" and r["N"]==N0:
            byS[r["S"]].append((r["M"], r["gflops"]))

    Ss = sorted(byS.keys())
    if not Ss: return

    bestG = []
    bestM = []
    for S in Ss:
        m,g = max(byS[S], key=lambda x: x[1])
        bestM.append(m); bestG.append(g)

    Sstar = Ss[max(range(len(Ss)), key=lambda i: bestG[i])]

    plt.figure()
    plt.plot(Ss, bestG, marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("S"); plt.ylabel("GFLOP/s")
    plt.title(f"blocked: best over M, P_real(S) at N={N0}, best S*={Sstar}")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(f"p8_blocked_P_vs_S_N{N0}.png", dpi=200)

def plot_best_vs_N():
    # пункт 9: P(N) для 4 лучших реализаций [1]
    Ns = sorted({r["N"] for r in rows})

    classic = {r["N"]: r["gflops"] for r in rows if r["alg"]=="classic_ijk"}
    tr_with = {r["N"]: r["gflops"] for r in rows if r["alg"]=="transposeB_ijk"}

    # best buffer over M per N
    best_buf = {}
    for N in Ns:
        cands = [r for r in rows if r["alg"]=="buffer_colB" and r["N"]==N]
        if cands:
            best_buf[N] = max(cands, key=lambda r: r["gflops"])["gflops"]

    # best blocked over (S,M) per N
    best_blk = {}
    for N in Ns:
        cands = [r for r in rows if r["alg"]=="blocked" and r["N"]==N]
        if cands:
            best_blk[N] = max(cands, key=lambda r: r["gflops"])["gflops"]

    x = [N for N in Ns if N in classic]
    plt.figure()
    plt.plot(x, [classic[N] for N in x], marker="o", label="classic_ijk")
    plt.plot(x, [tr_with[N] for N in x if N in tr_with], marker="o", label="transposeB (with transpose)")
    plt.plot(x, [best_buf[N] for N in x if N in best_buf], marker="o", label="buffer_colB (best M)")
    plt.plot(x, [best_blk[N] for N in x if N in best_blk], marker="o", label="blocked (best S,M)")
    plt.xlabel("N"); plt.ylabel("GFLOP/s")
    plt.title("Best implementations: P(N)")
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig("p9_best_P_vs_N.png", dpi=200)

plot_transpose_vs_classic()
plot_buffer_M(1024)
plot_blocked_S_bestM(1024)
plot_best_vs_N()
print("Saved: p6_transpose_vs_classic.png, p7_..., p8_..., p9_best_P_vs_N.png")
