"""DDAG E2 — A100 (Colab) runner.

Colab has no launch API, so this runs inside a Colab A100 session reached over a
cloudflared tunnel (see ideas/ddag/colab/README.md). It downloads the public Kaggle
inputs, then executes the same CRV capture kernel used on the T4.

Usage (inside the Colab VM):
    python a100_crv.py
"""

import os
import subprocess
import sys
from pathlib import Path

IN_ROOT = Path("/content/inputs")
OUT_ROOT = Path("/content/out")
KERNEL = Path("/content/crv.py")
DATASETS = ["ajjisan/causal-rcm-ckpts", "ajjisan/wan21-t5", "ajjisan/wan21-dit-vae"]


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kw)


def main():
    run([sys.executable, "-m", "pip", "install", "-q", "kaggle", "einops", "transformers",
         "sentencepiece", "ftfy", "regex", "imageio", "imageio-ffmpeg", "tqdm", "fvcore",
         "omegaconf", "attrs", "pyyaml", "iopath", "termcolor", "pynvml", "pandas", "loguru",
         "safetensors", "cloudpickle", "dill", "matplotlib", "packaging",
         "opencv-python-headless"], check=False)

    IN_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        run([sys.executable, "-m", "kaggle", "datasets", "download", "-d", ds,
             "-p", str(IN_ROOT), "--unzip"], check=False)

    os.environ["SA_INPUT_ROOT"] = str(IN_ROOT)
    os.environ["SA_OUT_ROOT"] = str(OUT_ROOT)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if not KERNEL.exists():
        raise SystemExit(f"place the CRV kernel at {KERNEL} (copy of ideas/ddag/jobs/crv/crv.py)")
    print("running CRV capture on A100...", flush=True)
    code = compile(KERNEL.read_text(encoding="utf-8"), str(KERNEL), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(KERNEL)})
    print("done; outputs under", OUT_ROOT / "results/selfaware_dit", flush=True)


if __name__ == "__main__":
    main()
