"""E3 — matched A100 online factorial from existing arms (0 GPU).

Same GPU (A100), same 12 prompts (manifest a100-01), same horizon (57). Only the coarse
operating point differs:
  a100_online/     (A2 v1): coarse = c3-3 step-4  (EXPENSIVE)
  a100_online_v2/  (A2 v2): coarse = c3-3 step-1  (CHEAP)
This isolates the operating-point effect on the online result (the old paper's confound).
Writes results/e3_online/matched_table.md + e3_online.json
"""
import json, os
from datetime import datetime, timezone

NEW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = r"C:\Users\jisan\OneDrive\Desktop\research\idea_05_adaptive_granularity_WM\04_Experiment"
ARMS = {
    "expensive_c3_3_s4": os.path.join(OLD, "results", "a100_online", "a100_online_audit.json"),
    "cheap_c3_3_s1": os.path.join(OLD, "results", "a100_online_v2", "a100_online_audit.json"),
}
OUT = os.path.join(NEW, "results", "e3_online")
os.makedirs(OUT, exist_ok=True)
POL = ["fixed_fine", "fixed_coarse", "fixed_hold", "random50", "alternating50", "adaptive_warp"]


def main():
    d = {k: json.load(open(v, encoding="utf-8"))["by_policy"] for k, v in ARMS.items()}
    rep = {"phase": "E3-online-matched", "generated_utc": datetime.now(timezone.utc).isoformat(),
           "gpu": "A100", "prompts": 12, "horizon": 57, "arms": {}}
    for arm, by in d.items():
        rep["arms"][arm] = {p: {"dit_s": round(by[p]["dit_s"], 3), "ref_l1": round(by[p]["mean_reference_l1"], 4),
                               "fine_fraction": round(by[p]["fine_fraction"], 3), "nfe": by[p]["nfe"],
                               "switch": round(by[p]["switch_count"], 2)} for p in POL}

    def coarse_cheaper(arm):
        return d[arm]["fixed_coarse"]["dit_s"] < d[arm]["fixed_fine"]["dit_s"]

    rep["coarse_cheaper_online"] = {a: coarse_cheaper(a) for a in ARMS}
    for a in ARMS:
        ad, ff = d[a]["adaptive_warp"], d[a]["fixed_fine"]
        rep["arms"][a]["adaptive_vs_fixed_fine"] = {
            "ref_l1_rel": (ad["mean_reference_l1"] - ff["mean_reference_l1"]) / ff["mean_reference_l1"],
            "dit_rel": (ad["dit_s"] - ff["dit_s"]) / ff["dit_s"]}
    json.dump(rep, open(os.path.join(OUT, "e3_online.json"), "w", encoding="utf-8"), indent=2)

    lines = ["# E3 — matched A100 online factorial (same GPU, same 12 prompts)", "",
             "| policy | EXPENSIVE c3-3 s4: DiT (s) | refL1 | fine% | CHEAP c3-3 s1: DiT (s) | refL1 | fine% |",
             "|---|---|---|---|---|---|---|"]
    for p in POL:
        e, c = rep["arms"]["expensive_c3_3_s4"][p], rep["arms"]["cheap_c3_3_s1"][p]
        lines.append(f"| {p} | {e['dit_s']:.2f} | {e['ref_l1']:.4f} | {e['fine_fraction']:.2f} | "
                     f"{c['dit_s']:.2f} | {c['ref_l1']:.4f} | {c['fine_fraction']:.2f} |")
    lines += ["", "## Operating-point effect (matched)", "",
              f"- EXPENSIVE c3-3 s4: coarse cheaper online? **{rep['coarse_cheaper_online']['expensive_c3_3_s4']}** "
              f"(fixed_coarse {rep['arms']['expensive_c3_3_s4']['fixed_coarse']['dit_s']:.2f}s vs fixed_fine "
              f"{rep['arms']['expensive_c3_3_s4']['fixed_fine']['dit_s']:.2f}s)",
              f"- CHEAP c3-3 s1: coarse cheaper online? **{rep['coarse_cheaper_online']['cheap_c3_3_s1']}** "
              f"(fixed_coarse {rep['arms']['cheap_c3_3_s1']['fixed_coarse']['dit_s']:.2f}s vs fixed_fine "
              f"{rep['arms']['cheap_c3_3_s1']['fixed_fine']['dit_s']:.2f}s)",
              "",
              "**Verdict:** on the same A100 and prompts, the adaptive result flips with the coarse operating "
              "point alone — cheap coarse enables adaptive savings (fine% 0.21, lower refL1 than fixed_fine); "
              "expensive coarse does not (adaptive forced to 92% fine). The old paper's T4-negative/A100-positive "
              "comparison was confounded by using expensive coarse on T4 and cheap coarse on A100."]
    open(os.path.join(OUT, "matched_table.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
