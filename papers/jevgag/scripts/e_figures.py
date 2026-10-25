"""Generate paper figures from E1/E2/E5/E7 results."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NEW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(NEW, "results")
FIG = os.path.join(NEW, "assets", "figures")
os.makedirs(FIG, exist_ok=True)

# ---- E1 break-even heatmap ----
m = json.load(open(os.path.join(R, "e1_breakeven", "breakeven_matrix.json")))["matrix"]
gpus = ["T4", "A100"]
fine = ["fine_s1", "fine_s2", "fine_s4"]
coarse = ["coarse_s1", "coarse_s2", "coarse_s4"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, gpu in zip(axes, gpus):
    grid = np.zeros((len(coarse), len(fine)))
    for r in m:
        if r["gpu"] != gpu:
            continue
        grid[coarse.index(r["coarse_mode"]), fine.index(r["fine_mode"])] = 1 if r["coarse_cheaper"] else 0
    ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(fine))); ax.set_xticklabels([f.replace("fine_", "") for f in fine])
    ax.set_yticks(range(len(coarse))); ax.set_yticklabels([c.replace("coarse_", "") for c in coarse])
    ax.set_xlabel("fine (c1-1 step)"); ax.set_ylabel("coarse (c3-3 step)")
    ax.set_title(f"{gpu}: coarse cheaper?")
    for i in range(len(coarse)):
        for j in range(len(fine)):
            ax.text(j, i, "Y" if grid[i, j] else "n", ha="center", va="center", fontsize=9)
fig.suptitle("E1 — break-even: coarse is cheaper only for some (fine,coarse) pairs and GPUs", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_e1_breakeven.png"), dpi=200); plt.close(fig)

# ---- E2 signal comparison ----
a = json.load(open(os.path.join(R, "e2_signal", "e2_audit.json")))
names = ["jev_difficulty", "jev_drift", "jev_motion"]
arms = ["text13", "text14", "image14", "hybrid14"]
x = np.arange(len(names)); w = 0.2
fig, ax = plt.subplots(figsize=(6.5, 3.4))
for k, arm in enumerate(arms):
    ax.bar(x + (k - 1.5) * w, [a["arms"][f"{arm}_{n}"]["auroc"] for n in names], w, label=arm)
ax.axhline(0.5, ls="--", c="grey", lw=1)
ax.set_xticks(x); ax.set_xticklabels([n.replace("jev_", "") for n in names])
ax.set_ylabel("AUROC"); ax.set_ylim(0, 1); ax.legend(fontsize=8)
ax.set_title("E2 — image/hybrid Jev do NOT beat text Jev (signal)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_e2_signal.png"), dpi=200); plt.close(fig)

# ---- E5 endpoint scatter ----
e5 = json.load(open(os.path.join(R, "e5_endpoint", "e5_audit.json")))
pol = e5["policies"]
fig, ax = plt.subplots(figsize=(5.2, 3.8))
ax.scatter([e5["jev_consistency"][p] for p in pol], [e5["clip_subject_consistency"][p] for p in pol], s=45)
for p in pol:
    ax.annotate(p, (e5["jev_consistency"][p], e5["clip_subject_consistency"][p]), fontsize=7)
ax.set_xlabel("Jev endpoint consistency (image)"); ax.set_ylabel("CLIP subject consistency")
ax.set_title(f"E5 — image-Jev endpoint agrees with CLIP (rho={e5['spearman_jev_consistency_vs_clip']:+.2f})")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_e5_endpoint.png"), dpi=200); plt.close(fig)

# ---- E7 frontier ----
e7 = json.load(open(os.path.join(R, "e7_confidence", "e7_audit.json")))
bs = [0.25, 0.5, 0.75]
fig, ax = plt.subplots(figsize=(5.6, 3.6))
for p in ["random", "difficulty_only", "confidence_gated", "oracle"]:
    ax.plot(bs, [e7["mean_regret"][p][str(b)] for b in bs], marker="o", label=p)
ax.set_xlabel("fine-decision budget"); ax.set_ylabel("mean reference regret")
ax.set_title("E7 — confidence-gating does not improve the frontier")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_e7_frontier.png"), dpi=200); plt.close(fig)

print("figures written to", FIG)
print(os.listdir(FIG))
