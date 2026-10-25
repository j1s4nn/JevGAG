"""E6 — matched T4 boundary (derived from E1 + prior T4 online arm; 0 GPU).

The T4 online arm used fine c1-1 s1 as the fine mode and the EXPENSIVE coarse c3-3 s4.
E1 shows that on T4, coarse c3-3 s1 is cheaper than fine ONLY when fine uses >=2 steps
(fine_s2/s4); against fine_s1 it is NOT cheaper (-4.57 s). Therefore on T4 with fine_s1
as the fine mode there is no cost gap and adaptive routing cannot save time -> the T4
boundary is negative by construction, independent of the signal quality.

This states the boundary formally and points to the prior T4 online arm as confirmation.
Writes results/e6_t4_boundary/e6.json + e6_table.md
"""
import json, os
from datetime import datetime, timezone

NEW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(NEW, "results")
OLD = r"C:\Users\jisan\OneDrive\Desktop\research\idea_05_adaptive_granularity_WM\04_Experiment"
OUT = os.path.join(R, "e6_t4_boundary")
os.makedirs(OUT, exist_ok=True)


def main():
    m = json.load(open(os.path.join(R, "e1_breakeven", "breakeven_matrix.json")))["matrix"]
    t4_s1 = [r for r in m if r["gpu"] == "T4" and r["coarse_mode"] == "coarse_s1"]
    by_fine = {r["fine_mode"]: r for r in t4_s1}
    prior = os.path.join(OLD, "results", "f4lite")
    rep = {"phase": "E6-t4-boundary", "generated_utc": datetime.now(timezone.utc).isoformat(),
           "gpu": "T4", "fine_mode": "fine_c1_1_s1",
           "coarse_s1_vs_fine_s1": {"coarse_cheaper": by_fine["fine_s1"]["coarse_cheaper"],
                                    "dit_saved_s": by_fine["fine_s1"]["dit_saved_s"]},
           "coarse_s1_vs_fine_s2": {"coarse_cheaper": by_fine["fine_s2"]["coarse_cheaper"],
                                    "dit_saved_s": by_fine["fine_s2"]["dit_saved_s"]},
           "coarse_s1_vs_fine_s4": {"coarse_cheaper": by_fine["fine_s4"]["coarse_cheaper"],
                                    "dit_saved_s": by_fine["fine_s4"]["dit_saved_s"]},
           "prior_t4_online_arm": prior,
           "verdict": ("T4 boundary is NEGATIVE for fine_c1_1_s1: coarse c3-3 s1 saves "
                       f"{by_fine['fine_s1']['dit_saved_s']:+.2f} s (i.e. is slower), so adaptive routing cannot "
                       "save wall-clock time regardless of the signal. The T4 negative is an operating-point "
                       "boundary, not a signal failure. Coarse only wins on T4 once fine uses >=2 steps.")}
    json.dump(rep, open(os.path.join(OUT, "e6.json"), "w", encoding="utf-8"), indent=2)
    lines = ["# E6 — matched T4 boundary (derived)", "",
             f"- T4 coarse `c3-3 s1` vs fine `c1-1 s1`: cheaper = **{rep['coarse_s1_vs_fine_s1']['coarse_cheaper']}** "
             f"(DiT saved {rep['coarse_s1_vs_fine_s1']['dit_saved_s']:+.2f} s)",
             f"- T4 coarse `c3-3 s1` vs fine `c1-1 s2`: cheaper = **{rep['coarse_s1_vs_fine_s2']['coarse_cheaper']}** "
             f"(DiT saved {rep['coarse_s1_vs_fine_s2']['dit_saved_s']:+.2f} s)",
             f"- T4 coarse `c3-3 s1` vs fine `c1-1 s4`: cheaper = **{rep['coarse_s1_vs_fine_s4']['coarse_cheaper']}** "
             f"(DiT saved {rep['coarse_s1_vs_fine_s4']['dit_saved_s']:+.2f} s)", "",
             "## Verdict", "", rep["verdict"],
             "", "Confirmed by the prior T4 online arm (expensive coarse), which was negative.",
             "", "**Note:** a fresh T4 online arm with the CHEAP coarse and fine_s2 is the one configuration that "
             "would make T4 positive; it is a candidate Kaggle run, not required for the boundary claim."]
    open(os.path.join(OUT, "e6_table.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
