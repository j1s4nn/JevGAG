"""E5 — image-Jev as an INDEPENDENT endpoint.

For each f4lite rollout video, ask Jev 1.14 (images) whether the subject is preserved
from the first to the last frame, and how consistent the subject is across the clip.
Then compare the per-policy ranking with CLIP subject consistency.

Reads (read-only): results/f4lite/videos/*.mp4
Writes: results/e5_endpoint/endpoint_answers.jsonl
"""
import base64, io, json, os, time, uuid
import cv2, numpy as np, requests
from PIL import Image

NEW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = r"C:\Users\jisan\OneDrive\Desktop\research\idea_05_adaptive_granularity_WM\04_Experiment"
VIDS = os.path.join(OLD, "results", "f4lite", "videos")
OUT = os.path.join(NEW, "results", "e5_endpoint")
os.makedirs(OUT, exist_ok=True)
ANSWERS = os.path.join(OUT, "endpoint_answers.jsonl")
ENDPOINT = "https://api.typesafe.ai/v1/systemone"; MODEL = "jev-1.14.0"

QUESTIONS = {
    "same_subject": {"type": "noul",
        "instructions": "Across the frames in `clip` (first to last), is the main subject the SAME individual or object throughout?",
        "criteria": {"true": "The same subject is present and recognisable in the last frame as in the first",
                     "false": "The subject changes identity, is replaced, or is lost"}},
    "consistency": {"type": "score",
        "instructions": "How consistent is the main subject's appearance and identity across the frames in `clip`?",
        "criteria": ["completely different", "mostly different", "partly consistent", "mostly consistent", "identical"]},
    "scene_drift": {"type": "noul",
        "instructions": "Does the background/scene drift noticeably from the first frame to the last in `clip`?",
        "criteria": {"true": "Clear background or layout change", "false": "Background stays the same"}},
}


def key():
    for line in open(os.path.join(os.path.expanduser("~"), ".kilo-automation", "typesafe.env"), encoding="utf-8"):
        if line.startswith("TYPESAFE_API_KEY="):
            return line.split("=", 1)[1].strip()


def frames(path):
    cap = cv2.VideoCapture(str(path)); out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release(); return np.asarray(out, dtype=np.uint8)


def b64(fr):
    img = Image.fromarray(fr)
    if max(img.size) > 1568:
        img.thumbnail((1568, 1568))
    b = io.BytesIO(); img.convert("RGB").save(b, "JPEG", quality=90)
    return base64.b64encode(b.getvalue()).decode("ascii")


def main():
    k = key()
    files = sorted(f for f in os.listdir(VIDS) if f.endswith(".mp4"))
    done = {json.loads(l)["video"] for l in open(ANSWERS, encoding="utf-8") if l.strip()} if os.path.exists(ANSWERS) else set()
    fh = open(ANSWERS, "a", encoding="utf-8")
    for fn in files:
        if fn in done:
            continue
        fr = frames(os.path.join(VIDS, fn))
        n = len(fr)
        picks = [0, n // 3, (2 * n) // 3, n - 1]
        assets, refs = {}, []
        for i in picks:
            r = str(uuid.uuid4()); refs.append(r); assets[r] = {"type": "image", "format": "jpg", "data_base64": b64(fr[i])}
        state = {"clip": [{"pos": p, "frame": r} for p, r in zip(picks, refs)]}
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
        prompt = fn.split("_")[0]
        policy = fn.split("_", 1)[1].rsplit(".", 1)[0]
        rec = {"video": fn, "prompt": prompt, "policy": policy, "frame_picks": picks,
               "latency_s": dt, "answer": ans, "error": err}
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
        print(f"{fn}: {'ok' if ans else 'ERR'} ({dt:.2f}s)")
    fh.close()


if __name__ == "__main__":
    main()
