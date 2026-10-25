"""E7 — confidence-gated routing (offline policy simulation, 0 GPU).

Question: does gating on Jev's own confidence (defer to fine when confidence is low)
improve the regret-vs-fine-budget frontier over difficulty-thresholding alone?

Per segment we have regret_fine / regret_coarse (from prior data) and, from E2,
the Jev difficulty score + confidence. A policy picks fine or coarse per segment;
achieved regret = regret_fine if fine else regret_coarse. Budget = fine fraction.
Writes results/e7_confidence/e7_audit.json + e7_table.md
"""
import json, os, random
from datetime import datetime, timezone

NEW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT14 = os.path.join(NEW, "results", "e2_signal", "text14_answers.jsonl")
OUT = os.path.join(NEW, "results", "e7_confidence")
os.makedirs(OUT, exist_ok=True)
BUDGETS = [0.25, 0.50, 0.75]
SEED = 20261026


def main():
    rows = [json.loads(l) for l in open(TEXT14, encoding="utf-8") if l.strip() and json.loads(l).get("answer")]
    segs = []
    for r in rows:
        a = r["answer"]["answers"]
        segs.append({"difficulty": a["difficulty"]["score"], "conf": a["difficulty"]["confidence"],
                     "rf": r["regret_fine"], "rc": r["regret_coarse"]})
    n = len(segs)
    rng = random.Random(SEED)

    def regret(order_fine):
        """order_fine: list of indices selected as fine (rest coarse)."""
        fine = set(order_fine)
        return sum((s["rf"] if i in fine else s["rc"]) for i, s in enumerate(segs)) / n

    def frontier(score_key):
        order = sorted(range(n), key=lambda i: -segs[i][score_key])
        out = {}
        for b in BUDGETS:
            k = max(1, round(b * n))
            out[b] = regret(order[:k])
        return out

    # signals to compare
    policies = {
        "difficulty_only": frontier("difficulty"),
        "confidence_gated": None,   # difficulty + (1-conf) tie-break
        "oracle": None,
    }
    # confidence-gated: primary = difficulty, secondary = (1 - confidence)
    order_cg = sorted(range(n), key=lambda i: -(segs[i]["difficulty"] + 0.5 * (1 - segs[i]["conf"])))
    policies["confidence_gated"] = {b: regret(order_cg[:max(1, round(b * n))]) for b in BUDGETS}
    # oracle: fine where regret_fine < regret_coarse
    oracle_fine = [i for i, s in enumerate(segs) if s["rf"] < s["rc"]]
    policies["oracle"] = {b: regret(oracle_fine) for b in BUDGETS}
    # random at each budget (mean over 200 draws)
    rnd = {}
    for b in BUDGETS:
        k = max(1, round(b * n))
        vals = []
        for _ in range(200):
            vals.append(regret(rng.sample(range(n), k)))
        rnd[b] = sum(vals) / len(vals)
    policies["random"] = rnd

    rep = {"phase": "E7-confidence-gate", "generated_utc": datetime.now(timezone.utc).isoformat(),
           "n": n, "budgets": BUDGETS, "mean_regret": policies,
           "oracle_fine_fraction": len(oracle_fine) / n}
    json.dump(rep, open(os.path.join(OUT, "e7_audit.json"), "w", encoding="utf-8"), indent=2)

    lines = ["# E7 — confidence-gated routing frontier (mean reference regret; lower is better)", "",
             "| policy | 25% fine | 50% fine | 75% fine |", "|---|---|---|---|"]
    for p in ["random", "difficulty_only", "confidence_gated", "oracle"]:
        v = policies[p]
        lines.append(f"| {p} | {v[0.25]:.4f} | {v[0.50]:.4f} | {v[0.75]:.4f} |")
    gain = {b: policies["difficulty_only"][b] - policies["confidence_gated"][b] for b in BUDGETS}
    lines += ["", "## confidence_gated minus difficulty_only (negative = better)", ""]
    for b in BUDGETS:
        lines.append(f"- {int(b*100)}% budget: {gain[b]:+.4f}")
    open(os.path.join(OUT, "e7_table.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
