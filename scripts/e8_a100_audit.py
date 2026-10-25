"""E8 (A100) audit — stability-gated recalibration vs threshold_only and full.

Reads results/e8_a100/results.jsonl. Controllers: threshold_only, full, recalibrate.
Reports mean reference L1, fine fraction, switch count, DiT seconds.
Writes results/e8_a100/e8_a100_table.md + e8_a100.json
"""
import json, os, statistics
from datetime import datetime, timezone

NEW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(NEW, "results", "e8_a100", "results.jsonl")
OUT = os.path.join(NEW, "results", "e8_a100")
CONTROLLERS = ["threshold_only", "full", "recalibrate"]


def main():
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("status") == "complete" and r.get("controller") in CONTROLLERS]
    by = {}
    for r in rows:
        by.setdefault(r["controller"], []).append(r)

    rep = {"phase": "E8-A100", "generated_utc": datetime.now(timezone.utc).isoformat(),
           "n_prompts": len(by.get("full", [])), "controllers": {}}
    for c in CONTROLLERS:
        rs = by.get(c, [])
        if not rs:
            continue
        rep["controllers"][c] = {
            "mean_reference_l1": round(statistics.fmean(r["mean_reference_l1"] for r in rs), 4),
            "mean_fine_fraction": round(statistics.fmean(r["fine_fraction"] for r in rs), 3),
            "mean_switch_count": round(statistics.fmean(r["switch_count"] for r in rs), 3),
            "mean_dit_s": round(statistics.fmean(r["timing"]["dit_s"] for r in rs), 3),
            "mean_total_s": round(statistics.fmean(r["timing"]["total_s"] for r in rs), 2),
            "mean_peak_vram_gb": round(statistics.fmean(r["timing"]["peak_vram_gb"] for r in rs), 2),
        }
    json.dump(rep, open(os.path.join(OUT, "e8_a100.json"), "w", encoding="utf-8"), indent=2)

    lines = [f"# E8 (A100) — stability-gated recalibration (n={rep['n_prompts']} prompts)", "",
             "| controller | mean refL1 | fine frac | switches | DiT s | total s | peak VRAM GB |",
             "|---|---|---|---|---|---|---|"]
    for c in CONTROLLERS:
        if c in rep["controllers"]:
            v = rep["controllers"][c]
            lines.append(f"| {c} | {v['mean_reference_l1']:.4f} | {v['mean_fine_fraction']:.2f} | "
                         f"{v['mean_switch_count']:.2f} | {v['mean_dit_s']:.2f} | {v['mean_total_s']:.1f} | {v['mean_peak_vram_gb']:.2f} |")
    if "full" in rep["controllers"] and "recalibrate" in rep["controllers"]:
        dq = rep["controllers"]["recalibrate"]["mean_reference_l1"] - rep["controllers"]["full"]["mean_reference_l1"]
        dc = rep["controllers"]["recalibrate"]["mean_dit_s"] - rep["controllers"]["full"]["mean_dit_s"]
        lines += ["", "## Verdict", "",
                  f"- recalibrate minus full: refL1 {dq:+.4f}, DiT {dc:+.2f} s.",
                  ("- **Recalibration does NOT help** (higher/equal refL1 at higher cost)."
                   if dq >= 0 else "- Recalibration reduces refL1.")]
    open(os.path.join(OUT, "e8_a100_table.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
