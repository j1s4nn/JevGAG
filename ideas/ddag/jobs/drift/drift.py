"""DDAG E3 — drift accumulation kernel (T4 / A100).

Question: does a coarse segment cause PERSISTENT drift, i.e. does its error PROPAGATE to
later segments? That is the premise of DALG (drift-aware lookahead).

For each prompt we run three chains over S segments at an 81-frame horizon (21 latent frames,
shared 3-frame coarse prefix, 6 segments of 3 latent frames):
  A  all fine        (F F F F F F)   -> baseline drift trajectory
  B  all coarse      (C C C C C C)   -> full-coarse trajectory
  C  one coarse then fine (C F F F F F) -> propagation test

Per segment we record the latent L1 to the c1-1 s4 reference at that segment's range.

Outputs <SA_OUT_ROOT>/results/drift/: measurements.npz + manifest.json

LIMIT = 8  # SA_LIMIT
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

LIMIT = 8  # SA_LIMIT
SEED = 17
HORIZON = 81
MID = [15 / 16, 5 / 6, 5 / 8]
N_REF_STEPS = 4
N_FINE_STEPS = 2
N_COARSE_STEPS = 2
RCM_DIR = Path("/tmp/rcm")
SEG = 3

OUT_DIR = Path(os.environ.get("SA_OUT_ROOT", "/kaggle/working")) / "results" / "drift"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MEAS = OUT_DIR / "measurements.npz"
MANIFEST = OUT_DIR / "manifest.json"

PROMPTS = [
    "a boat", "a book", "baseball stadium", "basement",
    "a bird flying over a snowy forest", "a waterfall crashing into a river",
    "A person is bungee jumping", "A person is riding a bicycle down a hill",
    "a train speeding through the countryside", "a campfire burning at night",
    "a bowl of soup", "a red sports car", "a white swan on a lake",
    "a busy city street at night", "a quiet library interior", "a sandy beach at sunset",
    "a bird soaring gracefully in the sky", "A person is canoeing or kayaking",
    "A person is dancing in the rain", "A person is skiing down a snowy slope",
    "A person is swimming in the ocean", "a wooden chair in a room",
    "an empty highway at dawn", "leaves falling from a tree",
    "a candle flickering on a table", "a kite flying in a clear blue sky",
    "a horse galloping across a field", "a boat sailing on a calm sea",
    "a robot arm assembling a circuit board", "a chef flipping a pancake",
    "a child blowing bubbles in a park", "rain falling on a city street",
]


def log(m):
    print(m, flush=True)


def find_under(root, filename):
    root = Path(root)
    return next((p for p in root.rglob(filename) if p.is_file()), None) if root.exists() else None


def install_deps():
    pkgs = ["einops", "transformers", "sentencepiece", "ftfy", "regex", "imageio", "imageio-ffmpeg",
            "tqdm", "fvcore", "omegaconf", "attrs", "pyyaml", "iopath", "termcolor", "pynvml",
            "pandas", "loguru", "safetensors", "cloudpickle", "dill", "matplotlib", "packaging",
            "opencv-python-headless"]
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


def main():
    t_start = time.time()
    manifest = {"run_id": datetime.now().strftime("%Y%m%d_%H%M%S"), "kernel": "ddag-e3-drift",
                "protocol": {"seed": SEED, "horizon": HORIZON, "limit": LIMIT, "seg": SEG,
                             "chains": ["A_all_fine", "B_all_coarse", "C_coarse_then_fine"]},
                "status": "running", "records": []}
    try:
        install_deps()
        manifest["gpu"] = torch.cuda.get_device_name() if torch.cuda.is_available() else "none"
        log(f"gpu={manifest['gpu']}")
        if not torch.cuda.is_available():
            raise RuntimeError("no GPU")
        if RCM_DIR.exists():
            shutil.rmtree(RCM_DIR)
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/NVlabs/rcm.git", str(RCM_DIR)], check=True, capture_output=True)
        manifest["upstream_rcm_commit"] = subprocess.check_output(["git", "-C", str(RCM_DIR), "rev-parse", "HEAD"], text=True).strip()

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
        n_seg = (latent_frames - SEG) // SEG
        manifest["latent"] = {"frames": int(latent_frames), "h": int(latent_h), "w": int(latent_w), "segments": int(n_seg)}
        log(f"latent_frames={latent_frames} segments={n_seg}")

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

        def run_segment(mode, boundary, noise_g, condition, gen_seed):
            net = get_net(mode)
            nsteps = N_COARSE_STEPS if mode == "coarse" else N_FINE_STEPS
            chunk = 3 if mode == "coarse" else 1
            spc = None if mode == "coarse" else [N_FINE_STEPS]
            steps = build_few_step_t_steps(nsteps, 1600, MID, "cuda")
            _, blocks, _ = make_block_pattern(4, latent_h, latent_w, 1, chunk, net.get_spatial_patch_size())
            counts = expand_steps_per_chunk(spc, blocks - 1, nsteps)
            sched = build_few_step_schedules_per_chunk(counts, spc, 1600, MID, [], "cuda")
            nz = noise_g.cuda()
            gen = torch.Generator(device="cuda").manual_seed(gen_seed)
            net.cuda()
            latent = causal_i2v_rollout_sampling(net, boundary.cuda(), nz, steps, sched, spc, condition, None, 1.0, chunk, False, generator=gen)
            fresh = latent[:, :, 1:].detach().cpu()
            net.cpu()
            torch.cuda.empty_cache()
            del latent, nz, sched, counts
            gc.collect()
            torch.cuda.empty_cache()
            return fresh

        N = min(LIMIT, len(PROMPTS))
        log(f"encoding {N} prompts...")
        EMBS = encode_prompts(PROMPTS[:N])

        per_chain = {c: [] for c in ("A_all_fine", "B_all_coarse", "C_coarse_then_fine")}
        recs = []
        t4 = time.time()
        for pi in range(N):
            prompt = PROMPTS[pi]
            tp = time.time()
            log(f"=== prompt {pi}: {prompt}")
            condition = {"crossattn_emb": EMBS[pi].to(**TENSOR_KWARGS)}
            gen0 = torch.Generator(device="cuda").manual_seed(SEED)
            noise = torch.randn(1, latent_c, latent_frames, latent_h, latent_w, dtype=torch.float32, device="cuda", generator=gen0).cpu()

            # reference (c1-1 s4) full rollout
            net = get_net("reference")
            steps = build_few_step_t_steps(N_REF_STEPS, 1600, MID, "cuda")
            _, blocks, _ = make_block_pattern(latent_frames, latent_h, latent_w, 1, 1, net.get_spatial_patch_size())
            counts = expand_steps_per_chunk(None, blocks, N_REF_STEPS)
            sched = build_few_step_schedules_per_chunk(counts, None, 1600, MID, [], "cuda")
            gen = torch.Generator(device="cuda").manual_seed(SEED + 1000)
            net.cuda()
            ref_latent = causal_rollout_sampling(net, noise.cuda(), steps, sched, None, condition, None, 1.0, 1, 1, False, generator=gen).detach().cpu()
            net.cpu(); torch.cuda.empty_cache(); del sched, counts; gc.collect(); torch.cuda.empty_cache()

            # shared coarse prefix
            net = get_net("coarse")
            steps = build_few_step_t_steps(N_COARSE_STEPS, 1600, MID, "cuda")
            _, blocks, _ = make_block_pattern(3, latent_h, latent_w, 3, 3, net.get_spatial_patch_size())
            counts = expand_steps_per_chunk(None, blocks, N_COARSE_STEPS)
            sched = build_few_step_schedules_per_chunk(counts, None, 1600, MID, [], "cuda")
            gen = torch.Generator(device="cuda").manual_seed(SEED + 2000)
            net.cuda()
            prefix = causal_rollout_sampling(net, noise[:, :, :3].cuda(), steps, sched, None, condition, None, 1.0, 3, 3, False, generator=gen).detach().cpu()
            net.cpu(); torch.cuda.empty_cache(); del sched, counts; gc.collect(); torch.cuda.empty_cache()

            def chain(mode_seq):
                b = prefix[:, :, -1:]
                l1s = []
                for g in range(1, n_seg + 1):
                    mode = mode_seq[g - 1]
                    fresh = run_segment(mode, b, noise[:, :, SEG * g:SEG * (g + 1)], condition, SEED + 2000 + g)
                    rng = slice(SEG + SEG * (g - 1), SEG + SEG * g)
                    ref = ref_latent[:, :, rng]
                    l1s.append(float((fresh.float() - ref.float()).abs().mean()))
                    b = fresh[:, :, -1:]
                return l1s

            for name, seq in (("A_all_fine", ["fine"] * n_seg),
                              ("B_all_coarse", ["coarse"] * n_seg),
                              ("C_coarse_then_fine", ["coarse"] + ["fine"] * (n_seg - 1))):
                l1s = chain(seq)
                per_chain[name].append(l1s)
                log(f"  {name}: " + " ".join(f"{v:.4f}" for v in l1s))
                recs.append({"prompt_idx": pi, "chain": name, "l1": l1s})

            del condition, noise, ref_latent, prefix
            gc.collect(); torch.cuda.empty_cache()
            manifest["records"].append({"prompt_idx": pi, "l1": {k: v[-1] for k, v in per_chain.items()}})
            log(f"  prompt {pi} done | {int(time.time() - tp)}s | elapsed {int(time.time() - t4)}s")

        arrs = {k: np.asarray(v, dtype=np.float32) for k, v in per_chain.items()}
        np.savez_compressed(MEAS, **arrs)
        manifest["n_prompts"] = N
        manifest["n_segments"] = int(n_seg)
        manifest["mean_drift_by_chain"] = {k: [float(x) for x in v.mean(axis=0)] for k, v in arrs.items()}
        manifest["status"] = "complete"
        manifest["elapsed_s"] = round(time.time() - t_start, 1)
        log("mean drift per segment: " + json.dumps(manifest["mean_drift_by_chain"]))
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
