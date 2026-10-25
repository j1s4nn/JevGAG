"""E2 step 1 — query Jev 1.14 with IMAGES (context frames) on the same 80 paired
segments used for the text-only baseline.

Paired comparison: identical segments, identical questions, only the state differs
(text caption -> raw frames). Resumable: re-running skips already-answered ids.

Reads (read-only, prior project):
  results/jev_signal/states.jsonl        (80 rows: prompt, horizon, group, context_frame_index, labels)
  results/f4lite/videos/p{idx}_fixed_fine.mp4          (horizon 57)
  results/rq4_monotonicity/videos/p{idx}_fixed_fine.mp4 (horizon 81)
Writes:
  results/e2_signal/image_states.jsonl
  results/e2_signal/image_answers.jsonl
"""
import base64
import io
import json
import os
import time
import uuid

import cv2
import numpy as np
import requests
from PIL import Image

NEW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = r"C:\Users\jisan\OneDrive\Desktop\research\idea_05_adaptive_granularity_WM\04_Experiment"
OUT = os.path.join(NEW, "results", "e2_signal")
os.makedirs(OUT, exist_ok=True)
STATES_IN = os.path.join(OLD, "results", "jev_signal", "states.jsonl")
STATES_OUT = os.path.join(OUT, "image_states.jsonl")
ANSWERS_OUT = os.path.join(OUT, "image_answers.jsonl")
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.14.0"

QUESTIONS = {
    "coarse_drift": {"type": "noul",
        "instructions": "Look at the frames in `context`. If the next few frames were generated as a coarse 3-frame chunk instead of one frame at a time, would they likely drift or become temporally inconsistent?",
        "criteria": {"true": "Fast/complex motion, occlusion, or a scene change a coarse chunk would blur",
                     "false": "Simple, slow, or static motion a coarse chunk would reproduce faithfully"}},
    "difficulty": {"type": "score",
        "instructions": "How difficult is it to predict the next few frames of this clip accurately?",
        "criteria": ["very easy", "easy", "moderate", "hard", "very hard"]},
    "motion": {"type": "choice",
        "instructions": "What kind of motion do the frames in `context` show?",
        "criteria": {"static": "Almost no motion", "moderate": "Some smooth motion",
                     "fast": "Fast or complex motion", "scene_change": "A cut or large scene change"}},
}


def load_key():
    p = os.path.join(os.path.expanduser("~"), ".kilo-automation", "typesafe.env")
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("TYPESAFE_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("TYPESAFE_API_KEY not found")


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.asarray(frames, dtype=np.uint8)


def video_path(row):
    idx = row["prompt_idx"]
    if row["horizon"] == 57:
        return os.path.join(OLD, "results", "f4lite", "videos", f"p{idx}_fixed_fine.mp4")
    return os.path.join(OLD, "results", "rq4_monotonicity", "videos", f"p{idx}_fixed_fine.mp4")


def jpeg_b64(frame):
    img = Image.fromarray(frame)
    if max(img.size) > 1568:
        img.thumbnail((1568, 1568))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_state(row, frames):
    cfi = row["context_frame_index"]
    idxs = [max(0, cfi - 2), max(0, cfi - 1), cfi]
    assets, refs = {}, []
    for k, i in enumerate(idxs):
        ref = str(uuid.uuid4())
        refs.append(ref)
        assets[ref] = {"type": "image", "format": "jpg", "data_base64": jpeg_b64(frames[i])}
    state = {"video_prompt": row["prompt"], "segment_index": row["segment_index"],
             "horizon_frames": row["horizon"],
             "recent_motion_magnitude": round(row["motion"], 3),
             "recent_temporal_warp": round(row["warp"], 3),
             "context": {"frames": [{"ago": 2, "frame": refs[0]}, {"ago": 1, "frame": refs[1]},
                                    {"ago": 0, "frame": refs[2]}]}}
    return state, assets, idxs


def main():
    key = load_key()
    rows = [json.loads(l) for l in open(STATES_IN, encoding="utf-8") if l.strip()]
    done = set()
    if os.path.exists(ANSWERS_OUT):
        done = {json.loads(l)["segment_id"] for l in open(ANSWERS_OUT, encoding="utf-8") if l.strip()}
    fh = open(ANSWERS_OUT, "a", encoding="utf-8")
    cache = {}
    lat = []
    for row in rows:
        sid = row["segment_id"]
        if sid in done:
            continue
        vp = video_path(row)
        if vp not in cache:
            cache[vp] = read_video(vp)
        frames = cache[vp]
        state, assets, idxs = build_state(row, frames)
        payload = {"model": MODEL, "state": state, "assets": assets, "questions": QUESTIONS}
        headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json",
                   "TypeSafe-Danger-Zone-Images": "true"}
        t0 = time.time()
        ans, err = None, None
        for attempt in range(4):
            try:
                r = requests.post(ENDPOINT, headers=headers, json=payload, timeout=60)
                if r.status_code in (429, 529):
                    time.sleep(2 ** attempt); continue
                r.raise_for_status(); ans = r.json(); break
            except Exception as e:
                err = f"{type(e).__name__}: {e}"; time.sleep(1 + attempt)
        dt = time.time() - t0; lat.append(dt)
        rec = {"segment_id": sid, "prompt_idx": row["prompt_idx"], "stratum": row["stratum"],
               "prompt": row["prompt"], "horizon": row["horizon"], "group": row["group"],
               "coarse_unsafe": row["coarse_unsafe"], "fine_gap": row["fine_gap"],
               "regret_fine": row["regret_fine"], "regret_coarse": row["regret_coarse"],
               "warp": row["warp"], "motion": row["motion"], "complexity": row["complexity"],
               "frame_indices": idxs, "latency_s": dt, "answer": ans, "error": err}
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
        if ans:
            a = ans["answers"]
            print(f"{sid}: drift={a['coarse_drift']['noul']:.2f} diff={a['difficulty']['score']:.2f} "
                  f"motion={a['motion']['choice']} unsafe={row['coarse_unsafe']} ({dt:.2f}s)")
        else:
            print(f"{sid}: ERROR {err}")
    fh.close()
    print(f"\ndone. mean latency {sum(lat)/len(lat):.2f}s over {len(lat)} new calls" if lat else "nothing new")


if __name__ == "__main__":
    main()
