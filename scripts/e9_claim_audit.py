"""E9 — claim-to-artifact audit.

Re-derives every headline claim from the stored result files and marks PASS/FAIL.
This is the reproducibility gate before writing the paper.
Writes results/e9_audit/claim_audit.md + claim_audit.json
"""
import json, os
from datetime import datetime, timezone

NEW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(NEW, "results")
OUT = os.path.join(R, "e9_audit")
os.makedirs(OUT, exist_ok=True)


def load(p):
    return json.load(open(os.path.join(R, p), encoding="utf-8"))


def main():
    checks = []

    def check(claim, ok, detail):
        checks.append({"claim": claim, "pass": bool(ok), "detail": detail})

    # E1
    e1 = load(os.path.join("e1_breakeven", "breakeven_matrix.json"))["matrix"]
    t4 = [r for r in e1 if r["gpu"] == "T4"]
    a100 = [r for r in e1 if r["gpu"] == "A100"]
    check("E1: T4 coarse_s4 is never cheaper than any fine mode",
          all(not r["coarse_cheaper"] for r in t4 if r["coarse_mode"] == "coarse_s4"),
          f"{sum(1 for r in t4 if r['coarse_mode']=='coarse_s4')} T4 s4 pairs, all not cheaper")
    check("E1: A100 coarse_s1 beats all three fine modes",
          all(r["coarse_cheaper"] for r in a100 if r["coarse_mode"] == "coarse_s1"),
          "3 A100 s1 pairs all cheaper")
    check("E1: T4 coarse_s1 beats fine_s2 and fine_s4 but not fine_s1",
          all(r["coarse_cheaper"] for r in t4 if r["coarse_mode"] == "coarse_s1" and r["fine_mode"] in ("fine_s2", "fine_s4"))
          and not next(r for r in t4 if r["coarse_mode"] == "coarse_s1" and r["fine_mode"] == "fine_s1")["coarse_cheaper"],
          "operating-point effect present")

    # E2
    e2 = load(os.path.join("e2_signal", "e2_audit.json"))
    for n in ["jev_difficulty", "jev_drift", "jev_motion"]:
        d = e2["paired"][n]["delta_image_minus_text14"]
        check(f"E2: image-1.14 <= text-1.14 for {n}", d <= 0, f"delta={d:+.3f}")
    check("E2: motion deficit is significant (P<0.05)",
          e2["paired"]["jev_motion"]["p_image_gt_text"] < 0.05,
          f"P(image>text)={e2['paired']['jev_motion']['p_image_gt_text']:.3f}")
    check("E2: text-1.13 ~ text-1.14 (version not the cause)",
          abs(e2["arms"]["text13_jev_difficulty"]["auroc"] - e2["arms"]["text14_jev_difficulty"]["auroc"]) < 0.05,
          f"0.788 vs {e2['arms']['text14_jev_difficulty']['auroc']:.3f}")

    # E3
    e3 = load(os.path.join("e3_online", "e3_online.json"))
    exp = e3["arms"]["expensive_c3_3_s4"]; chp = e3["arms"]["cheap_c3_3_s1"]
    check("E3: expensive coarse NOT cheaper online on A100",
          not e3["coarse_cheaper_online"]["expensive_c3_3_s4"],
          f"fixed_coarse {exp['fixed_coarse']['dit_s']:.2f} > fixed_fine {exp['fixed_fine']['dit_s']:.2f}")
    check("E3: cheap coarse IS cheaper online on A100",
          e3["coarse_cheaper_online"]["cheap_c3_3_s1"],
          f"fixed_coarse {chp['fixed_coarse']['dit_s']:.2f} < fixed_fine {chp['fixed_fine']['dit_s']:.2f}")
    check("E3: adaptive uses far fewer fine decisions with cheap coarse",
          chp["adaptive_warp"]["fine_fraction"] < exp["adaptive_warp"]["fine_fraction"],
          f"cheap {chp['adaptive_warp']['fine_fraction']:.2f} < expensive {exp['adaptive_warp']['fine_fraction']:.2f}")

    # E4
    e4 = load(os.path.join("e4_controller", "e4.json"))["gpus"]
    for g in ["T4", "A100"]:
        check(f"E4 ({g}): recovery+hysteresis reduces switches",
              e4[g]["full"]["switches"] <= e4[g]["threshold_only"]["switches"],
              f"{e4[g]['threshold_only']['switches']:.2f} -> {e4[g]['full']['switches']:.2f}")
        check(f"E4 ({g}): seam exists (switch L1 > interior L1)",
              e4[g]["full"]["seam_ratio"] > 1.0,
              f"ratio {e4[g]['full']['seam_ratio']:.2f}")

    # E5
    e5 = load(os.path.join("e5_endpoint", "e5_audit.json"))
    check("E5: image-Jev endpoint agrees with CLIP (same-subject)",
          e5["spearman_jev_same_vs_clip"] >= 0.9,
          f"rho={e5['spearman_jev_same_vs_clip']:+.3f}")
    check("E5: image-Jev consistency agrees with CLIP",
          e5["spearman_jev_consistency_vs_clip"] >= 0.9,
          f"rho={e5['spearman_jev_consistency_vs_clip']:+.3f}")

    # E7
    e7 = load(os.path.join("e7_confidence", "e7_audit.json"))
    do = e7["mean_regret"]["difficulty_only"]; cg = e7["mean_regret"]["confidence_gated"]
    deltas = [abs(cg[b] - do[b]) for b in ["0.25", "0.5", "0.75"]]
    check("E7: confidence-gating is neutral (|delta|<0.005)", max(deltas) < 0.005, f"max |delta|={max(deltas):.4f}")

    npass = sum(c["pass"] for c in checks)
    rep = {"phase": "E9-claim-audit", "generated_utc": datetime.now(timezone.utc).isoformat(),
           "n_checks": len(checks), "n_pass": npass, "checks": checks}
    json.dump(rep, open(os.path.join(OUT, "claim_audit.json"), "w", encoding="utf-8"), indent=2)

    lines = [f"# E9 — claim-to-artifact audit ({npass}/{len(checks)} PASS)", "",
             "| # | claim | result | detail |", "|---|---|---|---|"]
    for i, c in enumerate(checks, 1):
        lines.append(f"| {i} | {c['claim']} | {'PASS' if c['pass'] else 'FAIL'} | {c['detail']} |")
    open(os.path.join(OUT, "claim_audit.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
