"""DDAG E5 — A100 (Colab) online-policy runner.

Same pattern as a100_crv.py: unset stale offline env vars, pull inputs (public Kaggle ckpts +
HuggingFace VAE/T5), then execute the online capture kernel on the Colab A100.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

IN_ROOT = Path("/content/inputs")
OUT_ROOT = Path("/content/out")
KERNEL = Path("/content/online.py")


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kw)


def kaggle_cmd():
    if shutil.which("kaggle"):
        return ["kaggle"]
    return [sys.executable, "-m", "kaggle"]


def main():
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
    run(kaggle_cmd() + ["datasets", "download", "-d", "ajjisan/causal-rcm-ckpts",
                        "-p", str(IN_ROOT), "--unzip"], check=False)

    try:
        from huggingface_hub import snapshot_download
    except Exception:
        run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=False)
        from huggingface_hub import snapshot_download
    wan = IN_ROOT / "wan"
    wan.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(repo_id="Wan-AI/Wan2.1-T2V-1.3B", local_dir=str(wan),
                          allow_patterns=["Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth",
                                          "google/umt5-xxl/*"])
        print("hf snapshot ok", flush=True)
    except Exception as e:
        print("hf snapshot FAILED", e, flush=True)

    os.environ["SA_INPUT_ROOT"] = str(IN_ROOT)
    os.environ["SA_OUT_ROOT"] = str(OUT_ROOT)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if not KERNEL.exists():
        raise SystemExit(f"place the online kernel at {KERNEL}")
    print("running online capture on A100...", flush=True)
    code = compile(KERNEL.read_text(encoding="utf-8-sig"), str(KERNEL), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(KERNEL)})
    print("done; outputs under", OUT_ROOT / "results/online", flush=True)


if __name__ == "__main__":
    main()
