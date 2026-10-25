"""DDAG E2 — A100 (Colab) runner.

Colab has no launch API, so this runs inside a Colab A100 session reached over a
cloudflared tunnel (see ideas/ddag/colab/README.md). It downloads the public Kaggle
inputs, then executes the same CRV capture kernel used on the T4.

Usage (inside the Colab VM):
    python a100_crv.py
"""

import os
import shutil
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


def kaggle_cmd():
    if shutil.which("kaggle"):
        return ["kaggle"]
    return [sys.executable, "-m", "kaggle"]


def main():
    # the Colab kernel is persistent; a previous exec of the capture kernel may have left
    # HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE set, which blocks the input download below.
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.pop(k, None)

    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "kaggle", "einops", "transformers",
         "sentencepiece", "ftfy", "regex", "imageio", "imageio-ffmpeg", "tqdm", "fvcore",
         "omegaconf", "attrs", "pyyaml", "iopath", "termcolor", "pynvml", "pandas", "loguru",
         "safetensors", "cloudpickle", "dill", "matplotlib", "packaging",
         "opencv-python-headless"], check=False)
    if os.path.exists("/content/kaggle.json"):
        try:
            os.chmod("/content/kaggle.json", 0o600)
        except Exception:
            pass
        os.environ.setdefault("KAGGLE_CONFIG_DIR", "/content")

    IN_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    # checkpoints: public Kaggle dataset
    for ds in ["ajjisan/causal-rcm-ckpts"]:
        run(kaggle_cmd() + ["datasets", "download", "-d", ds, "-p", str(IN_ROOT), "--unzip"], check=False)

    # VAE + T5 + tokenizer: the Kaggle copies are private (403 outside our kernels), so pull
    # them from the public HuggingFace repo instead.
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=False)
        from huggingface_hub import snapshot_download
    repo = "Wan-AI/Wan2.1-T2V-1.3B"
    wan = IN_ROOT / "wan"
    wan.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo, local_dir=str(wan),
            allow_patterns=["Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth",
                            "google/umt5-xxl/*"],
        )
        print("hf snapshot ok ->", wan, flush=True)
    except Exception as e:
        print("hf snapshot FAILED", e, flush=True)

    os.environ["SA_INPUT_ROOT"] = str(IN_ROOT)
    os.environ["SA_OUT_ROOT"] = str(OUT_ROOT)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if not KERNEL.exists():
        raise SystemExit(f"place the CRV kernel at {KERNEL} (copy of ideas/ddag/jobs/crv/crv.py)")
    print("running CRV capture on A100...", flush=True)
    code = compile(KERNEL.read_text(encoding="utf-8-sig"), str(KERNEL), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(KERNEL)})
    print("done; outputs under", OUT_ROOT / "results/selfaware_dit", flush=True)


if __name__ == "__main__":
    main()
