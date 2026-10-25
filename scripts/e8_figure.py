"""E8 figure — recalibration is worse on both GPUs."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NEW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(NEW, "results")
FIG = os.path.join(NEW, "assets", "figures")
os.makedirs(FIG, exist_ok=True)

a = json.load(open(os.path.join(R, "e8_a100", "e8_a100.json")))["controllers"]
k = json.load(open(os.path.join(R, "e8_recalibration", "e8.json")))["policies"]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

# A100: refL1 and DiT per controller
ctrl = ["threshold_only", "full", "recalibrate"]
x = np.arange(len(ctrl))
ax = axes[0]
ax.bar(x - 0.2, [a[c]["mean_reference_l1"] for c in ctrl], 0.4, label="mean refL1")
ax.set_xticks(x); ax.set_xticklabels(ctrl, fontsize=8)
ax.set_ylabel("mean refL1"); ax.set_ylim(0.24, 0.30)
ax2 = ax.twinx()
ax2.plot(x + 0.2, [a[c]["mean_dit_s"] for c in ctrl], "o-", color="firebrick", label="DiT s")
ax2.set_ylabel("DiT s", color="firebrick")
ax.set_title("A100: recalibrate worse refL1 at ~2x DiT")

# T4: refL1 per policy
pol = ["fixed_coarse", "fixed_fine", "adaptive_warp", "recalibrate"]
ax = axes[1]
ax.bar(range(len(pol)), [k[p]["mean_reference_l1"] for p in pol])
ax.set_xticks(range(len(pol))); ax.set_xticklabels(pol, rotation=15, fontsize=8)
ax.set_ylabel("mean refL1"); ax.set_ylim(0.28, 0.33)
ax.set_title("T4: recalibrate no better than baselines")

fig.suptitle("E8 — stability-gated recalibration is NOT beneficial on either GPU", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_e8_recalibration.png"), dpi=200); plt.close(fig)
print("wrote", os.path.join(FIG, "fig_e8_recalibration.png"))
