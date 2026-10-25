"""Push a SelfAware-DiT Kaggle kernel, watch it tightly, and download outputs.

Usage:
  python ideas/selfaware-dit/scripts/kaggle_run.py --job capture --limit 2  --slug ajjisan/jevgag-selfaware-capture-smoke --out /tmp/sa_smoke
  python ideas/selfaware-dit/scripts/kaggle_run.py --job capture --limit 16 --slug ajjisan/jevgag-selfaware-capture       --out /tmp/sa_full

The LIMIT constant in capture.py is rewritten in a temp copy, so one source drives
smoke and full runs. Polls every 30 s and stops within ~1 min of completion.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
JOB_DIR = REPO / "ideas" / "selfaware-dit" / "jobs"


def run(cmd, cwd=None, check=True):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise SystemExit(f"command failed: {cmd}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=21600)
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()

    src = JOB_DIR / args.job
    stage = Path(tempfile.mkdtemp(prefix="sa_job_"))
    for f in src.iterdir():
        if f.is_file():
            shutil.copy(f, stage / f.name)

    if args.limit is not None:
        code = stage / "capture.py"
        txt = code.read_text(encoding="utf-8")
        txt2, nsub = re.subn(r"^LIMIT = .*# SA_LIMIT$", f"LIMIT = {args.limit}  # SA_LIMIT", txt, flags=re.M)
        if nsub == 0:
            raise SystemExit("LIMIT marker not found in capture.py")
        code.write_text(txt2, encoding="utf-8")

    meta = json.loads((stage / "kernel-metadata.json").read_text(encoding="utf-8"))
    owner, name = args.slug.split("/")
    meta["id"] = args.slug
    meta["title"] = name
    (stage / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[push] {args.slug} (limit={args.limit}) from {stage}")
    r = run(["python", "-m", "kaggle", "kernels", "push", "-p", str(stage)])
    slug = args.slug
    for line in (r.stdout + r.stderr).splitlines():
        m = re.search(r"kaggle\.com/code/([\w\-]+/[\w\-]+)", line)
        if m:
            slug = m.group(1)
    print(f"[push] slug={slug}")

    if args.no_wait:
        print("NOT WAITING")
        return

    start = time.time()
    while time.time() - start < args.timeout:
        s = run(["python", "-m", "kaggle", "kernels", "status", slug], check=False)
        raw = (s.stdout + s.stderr).strip().replace("\n", " ")
        el = int(time.time() - start)
        print(f"[{el:5d}s] {raw}")
        low = raw.lower()
        if "complete" in low:
            break
        if "error" in low or "fail" in low or "cancel" in low:
            print("[watch] kernel errored")
            break
        time.sleep(args.poll)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run(["python", "-m", "kaggle", "kernels", "output", slug, "-p", str(out)], check=False)
    print(f"[out] {out}")
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print(f"   {p.relative_to(out)}  {p.stat().st_size} B")


if __name__ == "__main__":
    main()
