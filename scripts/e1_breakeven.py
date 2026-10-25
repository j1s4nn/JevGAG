"""E1 — Break-even predictor for fine-vs-coarse temporal granularity.

The solver's core tool: given a fine mode (c1-1, S steps) and a coarse mode
(c3-3, S' steps) on a given GPU, predict whether coarse is cheaper in wall-clock
DiT time.

Rule: coarse is cheaper iff
    (s_per_nfe_coarse / s_per_nfe_fine) < (NFE_fine / NFE_coarse)
equivalently, iff  DiT_coarse < DiT_fine  at the same horizon.

Reads (read-only, prior project):
  results/f1_mode_cost/f1_mode_cost_audit.json        (T4)
  results/a100_mode_cost/a100_mode_cost_audit.json    (A100)
  results/a100_timing/a100_audit.json                 (A100 timing arm)
Writes:
  results/e1_breakeven/breakeven_matrix.json
  results/e1_breakeven/breakeven_matrix.csv
  results/e1_breakeven/breakeven_table.md
"""
import csv
import json
import os

NEW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = r"C:\Users\jisan\OneDrive\Desktop\research\idea_05_adaptive_granularity_WM\04_Experiment"
OUT = os.path.join(NEW, "results", "e1_breakeven")
os.makedirs(OUT, exist_ok=True)

SOURCES = {
    "T4": os.path.join(OLD, "results", "f1_mode_cost", "f1_mode_cost_audit.json"),
    "A100": os.path.join(OLD, "results", "a100_mode_cost", "a100_mode_cost_audit.json"),
}

FINE = {"fine_s1": "fine_c1_1_s1", "fine_s2": "fine_c1_1_s2", "fine_s4": "ref_c1_1_s4"}
COARSE = {"coarse_s1": "coarse_c3_3_s1", "coarse_s2": "coarse_c3_3_s2", "coarse_s4": "coarse_c3_3_s4"}


def load_gpu(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d["by_mode"]


def row(gpu, fm, cm, f, c):
    nfe_f, nfe_c = f["nfe"], c["nfe"]
    spn_f = f.get("s_per_nfe", f["dit_s"] / nfe_f)
    spn_c = c.get("s_per_nfe", c["dit_s"] / nfe_c)
    break_even = nfe_f / nfe_c
    ratio = spn_c / spn_f
    cheaper = c["dit_s"] < f["dit_s"]
    return {
        "gpu": gpu,
        "fine_mode": fm, "coarse_mode": cm,
        "fine_nfe": round(nfe_f, 2), "coarse_nfe": round(nfe_c, 2),
        "fine_dit_s": round(f["dit_s"], 3), "coarse_dit_s": round(c["dit_s"], 3),
        "fine_s_per_nfe": round(spn_f, 4), "coarse_s_per_nfe": round(spn_c, 4),
        "break_even_nfe_ratio": round(break_even, 3),
        "measured_per_nfe_ratio": round(ratio, 3),
        "coarse_cheaper": bool(cheaper),
        "dit_saved_s": round(f["dit_s"] - c["dit_s"], 3),
        "fine_ref_l1": round(f.get("ref_l1", float("nan")), 4),
        "coarse_ref_l1": round(c.get("ref_l1", float("nan")), 4),
    }


def main():
    matrix = []
    for gpu, path in SOURCES.items():
        by = load_gpu(path)
        for fm, fk in FINE.items():
            for cm, ck in COARSE.items():
                if fk in by and ck in by:
                    matrix.append(row(gpu, fm, cm, by[fk], by[ck]))

    with open(os.path.join(OUT, "breakeven_matrix.json"), "w", encoding="utf-8") as f:
        json.dump({"rule": "coarse_cheaper iff s_per_nfe_coarse/s_per_nfe_fine < NFE_fine/NFE_coarse",
                   "matrix": matrix}, f, indent=2)

    cols = list(matrix[0].keys())
    with open(os.path.join(OUT, "breakeven_matrix.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(matrix)

    # markdown table + headline
    lines = ["# E1 break-even matrix (matched: same modes on both GPUs)", "",
             "| GPU | fine | coarse | NFE f/c | break-even | measured ratio | coarse cheaper | DiT saved (s) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in matrix:
        lines.append("| {gpu} | {fine_mode} | {coarse_mode} | {fine_nfe}/{coarse_nfe} | "
                     "{break_even_nfe_ratio} | {measured_per_nfe_ratio} | {cc} | {dit_saved_s} |".format(
                         cc="**YES**" if r["coarse_cheaper"] else "no", **r))
    # headline: which fine step makes coarse win, per gpu x coarse
    lines += ["", "## Headline", ""]
    for gpu in SOURCES:
        for cm in COARSE:
            wins = [r["fine_mode"] for r in matrix if r["gpu"] == gpu and r["coarse_mode"] == cm and r["coarse_cheaper"]]
            lines.append(f"- **{gpu} / {cm}**: cheaper than fine modes -> {wins if wins else 'NONE (never cheaper)'}")
    with open(os.path.join(OUT, "breakeven_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
