"""DDAG qualitative figure — matched fine / coarse / repaired segments (T4).

Generates, for a few prompts, the matched fine and coarse segment from a shared boundary and the
blend-0.25 seam repair, decodes each to pixels, and writes PNG frames for the paper's qualitative grid.

Outputs <SA_OUT_ROOT>/results/qualitative/frames/*.png + manifest.json

LIMIT = 3  # SA_LIMIT
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import gc
import json
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

LIMIT = 3  # SA_LIMIT
SEED = 17
HORIZON = 57
MID = [15 / 16, 5 / 6, 5 / 8]
N_REF_STEPS = 4
N_FINE_STEPS = 2
N_COARSE_STEPS = 2
BLEND = 0.25
RCM_DIR = Path("/tmp/rcm")

OUT_DIR = Path(os.environ.get("SA_OUT_ROOT", "/kaggle/working")) / "results" / "qualitative"
FRAMES = OUT_DIR / "frames"
FRAMES.mkdir(parents=True, exist_ok=True)
MANIFEST = OUT_DIR / "manifest.json"

PROMPTS = ["a boat", "a bird flying over a snowy forest", "a waterfall crashing into a river",
           "A person is riding a bicycle down a hill"]


def log(m):
    print(m, flush=True)


def find_under(root, filename):
    root = Path(root)
    return next((p for p in root.rglob(filename) if p.is_file()), None) if root.exists() else None


def install_deps():
    pkgs = ["einops", "transformers", "sentencepiece", "ftfy", "regex", "imageio", "imageio-ffmpeg",
            "tqdm", "fvcore", "omegaconf", "attrs", "pyyaml", "iopath", "termcolor", "pynvml",
            "pandas", "loguru", "safetensors", "cloudpickle", "dill", "matplotlib", "packaging",
            "opencv-python-headless", "pillow"]
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *pkgs], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"pip install failed: {r.stderr[-400:]}")


def patch_file(path, repl):
    path = Path(path)
    s = path.read_text(encoding="utf-8")
    for o, n in repl:
        if o not in s:
            raise RuntimeError(f"missing patch target in {path.name}: {o[:70]}")
        s = s.replace(o, n)
    path.write_text(s, encoding="utf-8")


def save_frames(frames_u8, name):
    from PIL import Image
    idx = [0, frames_u8.shape[0] // 2, frames_u8.shape[0] - 1]
    for k, i in enumerate(idx):
        Image.fromarray(frames_u8[i]).save(FRAMES / f"{name}_f{k}.png")
    return [int(i) for i in idx]


def main():
    t0 = time.time()
    manifest = {"run_id": datetime.now().strftime("%Y%m%d_%H%M%S"), "kernel": "ddag-qualitative",
                "status": "running", "frames": []}
    try:
        install_deps()
        manifest["gpu"] = torch.cuda.get_device_name() if torch.cuda.is_available() else "none"
        if not torch.cuda.is_available():
            raise RuntimeError("no GPU")
        if RCM_DIR.exists():
            shutil.rmtree(RCM_DIR)
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/NVlabs/rcm.git", str(RCM_DIR)], check=True, capture_output=True)

        inp = Path(os.environ.get("SA_INPUT_ROOT", "/kaggle/input"))
        paths = {
            "fine": find_under(inp, "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step2.pt"),
            "reference": find_under(inp, "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step4.pt"),
            "coarse": find_under(inp, "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c3-3_step2.pt"),
            "vae": find_under(inp, "Wan2.1_VAE.pth"),
            "t5": find_under(inp, "models_t5_umt5-xxl-enc-bf16.pth"),
            "tokenizer": find_under(inp, "tokenizer.json"),
        }
        if paths["tokenizer"]:
            paths["tokenizer"] = paths["tokenizer"].parent
        missing = [k for k, v in paths.items() if v is None]
        if missing:
            raise FileNotFoundError(f"missing: {missing}")

        patch_file(RCM_DIR / "rcm/inference/wan2pt1_t2v_causal_infer.py", [
            ('TENSOR_KWARGS = {"device": "cuda", "dtype": torch.bfloat16}', 'TENSOR_KWARGS = {"device": "cuda", "dtype": torch.float16}'),
            (".to(dtype=torch.bfloat16).cuda()", ".to(dtype=torch.float16).cuda()"),
        ])
        patch_file(RCM_DIR / "rcm/utils/umt5.py", [
            ("dtype=torch.bfloat16,", "dtype=torch.float16,"),
            ('tokenizer_path="google/umt5-xxl",', f'tokenizer_path="{paths["tokenizer"]}",'),
        ])
        sys.path.insert(0, str(RCM_DIR))
        from imaginaire.lazy_config import instantiate
        from rcm.inference.wan2pt1_t2v_causal_infer import (
            DIT_CONFIGS, TENSOR_KWARGS, build_few_step_t_steps, build_few_step_schedules_per_chunk,
            expand_steps_per_chunk, make_block_pattern, load_dit_weights,
            causal_rollout_sampling, causal_i2v_rollout_sampling,
        )
        from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
        from rcm.utils.model_utils import init_weights_on_device
        from rcm.utils.umt5 import clear_umt5_memory, umt5_xxl, HuggingfaceTokenizer
        from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
        from imaginaire.utils.easy_io import easy_io
        TENSOR_KWARGS["dtype"] = torch.float16

        tokenizer = Wan2pt1VAEInterface(vae_pth=str(paths["vae"]))
        width, height = VIDEO_RES_SIZE_INFO["480p"]["16:9"]
        latent_frames = tokenizer.get_latent_num_frames(HORIZON)
        latent_h = height // tokenizer.spatial_compression_factor
        latent_w = width // tokenizer.spatial_compression_factor
        latent_c = tokenizer.latent_ch

        def encode_prompts(prompts):
            model = umt5_xxl(encoder_only=True).eval().requires_grad_(False).to(dtype=torch.float16)
            ckpt = easy_io.load(str(paths["t5"]), backend_args=None, file_format="pt", map_location="cuda", weights_only=False)
            for k in list(ckpt.keys()):
                if torch.is_tensor(ckpt[k]):
                    ckpt[k] = ckpt[k].to(torch.float16)
            model.load_state_dict(ckpt, strict=False, assign=True)
            del ckpt
            gc.collect()
            tok = HuggingfaceTokenizer(name=str(paths["tokenizer"]), seq_len=512, clean="whitespace")
            out = []
            for p in prompts:
                ids, mask = tok(p, return_mask=True, add_special_tokens=True)
                ids, mask = ids.cuda(), mask.cuda()
                ctx = model(ids, mask)
                lengths = mask.gt(0).sum(dim=1).long()
                padded = []
                for item, ln in zip(ctx, lengths):
                    ln = int(ln)
                    padded.append(item[:512] if ln > 512 else torch.cat([item[:ln], torch.zeros(512 - ln, item.shape[1], device=item.device, dtype=item.dtype)], dim=0))
                out.append(torch.stack(padded).to(torch.float16).cpu())
                del ids, mask, ctx, padded
                torch.cuda.empty_cache()
            del model, tok
            clear_umt5_memory()
            gc.collect()
            torch.cuda.empty_cache()
            return out

        NETS = {}

        def get_net(key):
            if key not in NETS:
                with init_weights_on_device():
                    net = instantiate(DIT_CONFIGS["1.3B"]).eval()
                load_dit_weights(net, str(paths[key]))
                net.to(**TENSOR_KWARGS).cpu()
                torch.cuda.empty_cache()
                NETS[key] = net
            return NETS[key]

        def decode(latent):
            video = tokenizer.decode(latent.float().cuda())
            px = (1 + video.float().cpu().clamp(-1, 1)) / 2
            if px.ndim == 5:
                px = px[0]
            frames = (px.permute(1, 2, 3, 0).numpy() * 255).astype(np.uint8)
            del video
            torch.cuda.empty_cache()
            return frames

        N = min(LIMIT, len(PROMPTS))
        EMBS = encode_prompts(PROMPTS[:N])

        for pi in range(N):
            prompt = PROMPTS[pi]
            log(f"=== prompt {pi}: {prompt}")
            condition = {"crossattn_emb": EMBS[pi].to(**TENSOR_KWARGS)}
            gen0 = torch.Generator(device="cuda").manual_seed(SEED)
            noise = torch.randn(1, latent_c, latent_frames, latent_h, latent_w, dtype=torch.float32, device="cuda", generator=gen0).cpu()

            net = get_net("coarse")
            steps = build_few_step_t_steps(N_COARSE_STEPS, 1600, MID, "cuda")
            _, blocks, _ = make_block_pattern(3, latent_h, latent_w, 3, 3, net.get_spatial_patch_size())
            counts = expand_steps_per_chunk(None, blocks, N_COARSE_STEPS)
            sched = build_few_step_schedules_per_chunk(counts, None, 1600, MID, [], "cuda")
            gen = torch.Generator(device="cuda").manual_seed(SEED + 2000)
            net.cuda()
            prefix = causal_rollout_sampling(net, noise[:, :, :3].cuda(), steps, sched, None, condition, None, 1.0, 3, 3, False, generator=gen).detach().cpu()
            net.cpu(); torch.cuda.empty_cache(); del sched, counts; gc.collect(); torch.cuda.empty_cache()
            boundary = prefix[:, :, -1:]

            def run_segment(mode, g):
                net = get_net(mode)
                nsteps = N_COARSE_STEPS if mode == "coarse" else N_FINE_STEPS
                chunk = 3 if mode == "coarse" else 1
                spc = None if mode == "coarse" else [N_FINE_STEPS]
                steps = build_few_step_t_steps(nsteps, 1600, MID, "cuda")
                _, blocks, _ = make_block_pattern(4, latent_h, latent_w, 1, chunk, net.get_spatial_patch_size())
                counts = expand_steps_per_chunk(spc, blocks - 1, nsteps)
                sched = build_few_step_schedules_per_chunk(counts, spc, 1600, MID, [], "cuda")
                nz = noise[:, :, 3 * g:3 * (g + 1)].cuda()
                gen = torch.Generator(device="cuda").manual_seed(SEED + 2000 + g)
                net.cuda()
                latent = causal_i2v_rollout_sampling(net, boundary.cuda(), nz, steps, sched, spc, condition, None, 1.0, chunk, False, generator=gen)
                fresh = latent[:, :, 1:].detach().cpu()
                net.cpu(); torch.cuda.empty_cache(); del latent, nz, sched, counts; gc.collect(); torch.cuda.empty_cache()
                return fresh

            for g in (1, 2):
                F = run_segment("fine", g)
                C = run_segment("coarse", g)
                bf = boundary.float()
                Cb = C.clone()
                Cb[:, :, 0:1] = (1 - BLEND) * C[:, :, 0:1].float() + BLEND * bf
                for tag, lat in (("fine", F), ("coarse", C), ("repair", Cb)):
                    fr = decode(lat)
                    idx = save_frames(fr, f"p{pi}_s{g}_{tag}")
                    manifest["frames"].append({"prompt": prompt, "segment": g, "variant": tag, "frames": idx})
                    del fr
                boundary = F[:, :, -1:]
                del F, C, Cb, bf
                gc.collect(); torch.cuda.empty_cache()
            del condition, noise, prefix, boundary
            gc.collect(); torch.cuda.empty_cache()
            log(f"  prompt {pi} done")

        manifest["status"] = "complete"
        manifest["elapsed_s"] = round(time.time() - t0, 1)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["traceback"] = traceback.format_exc()
        log(traceback.format_exc())
    finally:
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        log(f"manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
