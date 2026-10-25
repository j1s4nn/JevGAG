"""E2 step 5 — HYBRID arm: caption + frames in one state, Jev 1.14.
Tests whether the image deficit is recovered by keeping the caption alongside frames.
Writes results/e2_signal/hybrid_answers.jsonl
"""
import base64, io, json, os, time, uuid
import cv2, numpy as np, requests
from PIL import Image

NEW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = r"C:\Users\jisan\OneDrive\Desktop\research\idea_05_adaptive_granularity_WM\04_Experiment"
OUT = os.path.join(NEW, "results", "e2_signal")
STATES = os.path.join(OLD, "results", "jev_signal", "states.jsonl")
ANSWERS = os.path.join(OUT, "hybrid_answers.jsonl")
ENDPOINT = "https://api.typesafe.ai/v1/systemone"; MODEL = "jev-1.14.0"
QUESTIONS = {
    "coarse_drift": {"type": "noul",
        "instructions": "Given the context description in `context_caption` and the frames in `context`, would generating the next few frames with a coarse 3-frame-chunk method likely drift or become temporally inconsistent compared with a frame-wise method?",
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


def key():
    for line in open(os.path.join(os.path.expanduser("~"), ".kilo-automation", "typesafe.env"), encoding="utf-8"):
        if line.startswith("TYPESAFE_API_KEY="):
            return line.split("=", 1)[1].strip()


def vpath(row):
    idx = row["prompt_idx"]
    return os.path.join(OLD, "results", "f4lite" if row["horizon"] == 57 else "rq4_monotonicity",
                        "videos", f"p{idx}_fixed_fine.mp4")


def frames_of(path):
    cap = cv2.VideoCapture(str(path)); out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release(); return np.asarray(out, dtype=np.uint8)


def b64(frame):
    img = Image.fromarray(frame)
    if max(img.size) > 1568:
        img.thumbnail((1568, 1568))
    b = io.BytesIO(); img.convert("RGB").save(b, "JPEG", quality=90)
    return base64.b64encode(b.getvalue()).decode("ascii")


def main():
    k = key()
    rows = [json.loads(l) for l in open(STATES, encoding="utf-8") if l.strip()]
    done = {json.loads(l)["segment_id"] for l in open(ANSWERS, encoding="utf-8") if l.strip()} if os.path.exists(ANSWERS) else set()
    fh = open(ANSWERS, "a", encoding="utf-8"); cache = {}
    for row in rows:
        if row["segment_id"] in done:
            continue
        vp = vpath(row)
        if vp not in cache:
            cache[vp] = frames_of(vp)
        fr = cache[vp]; cfi = row["context_frame_index"]
        idxs = [max(0, cfi - 2), max(0, cfi - 1), cfi]
        assets, refs = {}, []
        for i in idxs:
            r = str(uuid.uuid4()); refs.append(r); assets[r] = {"type": "image", "format": "jpg", "data_base64": b64(fr[i])}
        state = {"video_prompt": row["prompt"], "context_caption": row["caption"],
                 "segment_index": row["segment_index"], "horizon_frames": row["horizon"],
                 "recent_motion_magnitude": round(row["motion"], 3),
                 "recent_temporal_warp": round(row["warp"], 3),
                 "context": {"frames": [{"ago": 2, "frame": refs[0]}, {"ago": 1, "frame": refs[1]}, {"ago": 0, "frame": refs[2]}]}}
        payload = {"model": MODEL, "state": state, "assets": assets, "questions": QUESTIONS}
        t0 = time.time(); ans = err = None
        for a in range(4):
            try:
                r = requests.post(ENDPOINT, headers={"Authorization": "Bearer " + k, "Content-Type": "application/json",
                                  "TypeSafe-Danger-Zone-Images": "true"}, json=payload, timeout=60)
                if r.status_code in (429, 529):
                    time.sleep(2 ** a); continue
                r.raise_for_status(); ans = r.json(); break
            except Exception as e:
                err = f"{type(e).__name__}: {e}"; time.sleep(1 + a)
        dt = time.time() - t0
        rec = {**{kk: row[kk] for kk in ("segment_id", "prompt_idx", "stratum", "prompt", "horizon", "group",
                                          "coarse_unsafe", "fine_gap", "regret_fine", "regret_coarse", "warp", "motion", "complexity")},
               "model": MODEL, "latency_s": dt, "answer": ans, "error": err}
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
        print(f"{row['segment_id']}: {'ok' if ans else 'ERR'} ({dt:.2f}s)")
    fh.close()


if __name__ == "__main__":
    main()
