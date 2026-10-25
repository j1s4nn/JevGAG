"""E2 step 3 — re-run the TEXT state on Jev 1.14 (the old text baseline used 1.13).

This removes the model-version confound from the image-vs-text comparison: after this,
both arms use jev-1.14.0 and differ only in the state representation (caption vs frames).

Reads (read-only): prior results/jev_signal/states.jsonl
Writes: results/e2_signal/text14_answers.jsonl
"""
import json
import os
import time

import requests

NEW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = r"C:\Users\jisan\OneDrive\Desktop\research\idea_05_adaptive_granularity_WM\04_Experiment"
OUT = os.path.join(NEW, "results", "e2_signal")
STATES = os.path.join(OLD, "results", "jev_signal", "states.jsonl")
ANSWERS = os.path.join(OUT, "text14_answers.jsonl")
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.14.0"

QUESTIONS = {
    "coarse_drift": {"type": "noul",
        "instructions": "Given this video context, would generating the next few frames with a coarse 3-frame-chunk method likely drift or become temporally inconsistent compared with a frame-wise method?",
        "criteria": {"true": "Fast or complex motion, occlusion, or a scene change that a coarse chunk would blur or smear",
                     "false": "Simple, slow, or static motion that a coarse chunk would reproduce faithfully"}},
    "difficulty": {"type": "score",
        "instructions": "How difficult is it to predict the next few frames of this video accurately?",
        "criteria": ["very easy", "easy", "moderate", "hard", "very hard"]},
    "motion": {"type": "choice",
        "instructions": "What kind of motion best describes this context?",
        "criteria": {"static": "Almost no motion", "moderate": "Some smooth motion",
                     "fast": "Fast or complex motion", "scene_change": "A cut or large scene change"}},
}


def load_key():
    p = os.path.join(os.path.expanduser("~"), ".kilo-automation", "typesafe.env")
    for line in open(p, encoding="utf-8"):
        if line.startswith("TYPESAFE_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no key")


def state_of(row):
    return {"video_prompt": row["prompt"], "context_caption": row["caption"],
            "segment_index": row["segment_index"], "horizon_frames": row["horizon"],
            "recent_motion_magnitude": round(row["motion"], 3),
            "recent_temporal_warp": round(row["warp"], 3)}


def main():
    key = load_key()
    rows = [json.loads(l) for l in open(STATES, encoding="utf-8") if l.strip()]
    done = {json.loads(l)["segment_id"] for l in open(ANSWERS, encoding="utf-8") if l.strip()} if os.path.exists(ANSWERS) else set()
    fh = open(ANSWERS, "a", encoding="utf-8")
    lat = []
    for row in rows:
        if row["segment_id"] in done:
            continue
        payload = {"model": MODEL, "state": state_of(row), "questions": QUESTIONS}
        t0 = time.time(); ans = err = None
        for a in range(4):
            try:
                r = requests.post(ENDPOINT, headers={"Authorization": "Bearer " + key,
                                  "Content-Type": "application/json"}, json=payload, timeout=40)
                if r.status_code in (429, 529):
                    time.sleep(2 ** a); continue
                r.raise_for_status(); ans = r.json(); break
            except Exception as e:
                err = f"{type(e).__name__}: {e}"; time.sleep(1 + a)
        dt = time.time() - t0; lat.append(dt)
        rec = {"segment_id": row["segment_id"], "prompt_idx": row["prompt_idx"], "stratum": row["stratum"],
               "prompt": row["prompt"], "horizon": row["horizon"], "group": row["group"],
               "coarse_unsafe": row["coarse_unsafe"], "fine_gap": row["fine_gap"],
               "regret_fine": row["regret_fine"], "regret_coarse": row["regret_coarse"],
               "warp": row["warp"], "motion": row["motion"], "complexity": row["complexity"],
               "model": MODEL, "latency_s": dt, "answer": ans, "error": err}
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
        print(f"{row['segment_id']}: {'ok' if ans else 'ERR ' + str(err)} ({dt:.2f}s)")
    fh.close()
    print(f"done, mean {sum(lat)/len(lat):.2f}s" if lat else "nothing new")


if __name__ == "__main__":
    main()
