"""SeamControl E2 kernel — measure the mode-switch boundary seam and test a
training-free repair, in the matched design (Kaggle T4).

For every prompt and segment:
  - boundary b = last latent frame of the previous (fine) segment, or the coarse prefix.
  - generate the segment in fine mode (F, 3 latent frames) and coarse mode (C, 3 frames)
    from the SAME boundary -> the boundary discontinuity L1(b, first frame) is the seam a
    switch to that mode produces.
  - repair variants applied to C's first frame:
        none | blend(a in {0.25,0.5,0.75}) : x1 <- (1-a)x1 + a*b | anchor : x1 <- b
  - for each variant record: boundary seam and fidelity (latent L1 to the c1-1 s4 reference).

Outputs /kaggle/working/results/seamcontrol/: measurements.npz + manifest.json

LIMIT = 16  # SA_LIMIT
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

LIMIT = 16  # SA_LIMIT
SEED = 17
HORIZON = 57
MID = [15 / 16, 5 / 6, 5 / 8]
N_REF_STEPS = 4
N_FINE_STEPS = 2
N_COARSE_STEPS = 2
RCM_DIR = Path("/tmp/rcm")
BLENDS = [0.25, 0.5, 0.75]

OUT_DIR = Path("/kaggle/working/results/seamcontrol")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MEAS = OUT_DIR / "measurements.npz"
MANIFEST = OUT_DIR / "manifest.json"

PROMPTS = [
    "a boat", "a book", "a bowl of soup", "a red sports car", "a white swan on a lake",
    "baseball stadium", "basement", "a busy city street at night", "a quiet library interior",
    "a sandy beach at sunset", "a bird flying over a snowy forest", "a bird soaring gracefully in the sky",
    "a waterfall crashing into a river", "a train speeding through the countryside", "a campfire burning at night",
    "A person is bungee jumping", "A person is canoeing or kayaking", "A person is riding a bicycle down a hill",
    "A person is dancing in the rain", "A person is skiing down a snowy slope",
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
        raise RuntimeError(f"pip install failed: {r.stderr[-500:]}")


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
    manifest = {"run_id": datetime.now().strftime("%Y%m%d_%H%M%S"), "kernel": "seamcontrol-repair",
                "protocol": {"seed": SEED, "horizon": HORIZON, "limit": LIMIT, "blends": BLENDS,
                             "fine_ckpt": "c1-1_step2", "coarse_ckpt": "c3-3_step2", "ref_ckpt": "c1-1_step4"},
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

        inp = Path("/kaggle/input")
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
        if any(v is None for v in paths.values()):
            raise FileNotFoundError({k: str(v) for k, v in paths.items()})

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

        N = min(LIMIT, len(PROMPTS))
        log(f"encoding {N} prompts...")
        EMBS = encode_prompts(PROMPTS[:N])

        rows = []  # per (prompt, segment, variant)
        t4 = time.time()
        for pi in range(N):
            prompt = PROMPTS[pi]
            tp = time.time()
            log(f"=== prompt {pi}: {prompt}")
            condition = {"crossattn_emb": EMBS[pi].to(**TENSOR_KWARGS)}
            gen0 = torch.Generator(device="cuda").manual_seed(SEED)
            noise = torch.randn(1, latent_c, latent_frames, latent_h, latent_w, dtype=torch.float32, device="cuda", generator=gen0).cpu()

            # reference rollout
            net = get_net("reference")
            steps = build_few_step_t_steps(N_REF_STEPS, 1600, MID, "cuda")
            _, blocks, _ = make_block_pattern(latent_frames, latent_h, latent_w, 1, 1, net.get_spatial_patch_size())
            counts = expand_steps_per_chunk(None, blocks, N_REF_STEPS)
            sched = build_few_step_schedules_per_chunk(counts, None, 1600, MID, [], "cuda")
            gen = torch.Generator(device="cuda").manual_seed(SEED + 1000)
            net.cuda()
            ref_latent = causal_rollout_sampling(net, noise.cuda(), steps, sched, None, condition, None, 1.0, 1, 1, False, generator=gen).detach().cpu()
            net.cpu(); torch.cuda.empty_cache(); del sched, counts; gc.collect(); torch.cuda.empty_cache()

            # coarse prefix
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

            def run_segment(mode):
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

            for g in range(1, 5):
                F = run_segment("fine")
                C = run_segment("coarse")
                b = boundary  # [1,C,1,H,W]
                ref_seg = ref_latent[:, :, 3 + 3 * (g - 1):3 + 3 * g]
                bf = b.float()

                def boundary_seam(seg):
                    return float((seg[:, :, 0:1].float() - bf).abs().mean())

                def interior(seg):
                    return float((seg[:, :, 1:].float() - seg[:, :, :-1].float()).abs().mean())

                def fidelity(seg):
                    return float((seg.float() - ref_seg.float()).abs().mean())

                base = {"prompt_idx": pi, "prompt": prompt, "segment": g,
                        "seam_fine": boundary_seam(F), "interior_fine": interior(F), "fid_fine": fidelity(F),
                        "seam_coarse": boundary_seam(C), "interior_coarse": interior(C), "fid_coarse": fidelity(C)}

                variants = {"none": C.clone()}
                for a in BLENDS:
                    Cb = C.clone()
                    Cb[:, :, 0:1] = (1 - a) * C[:, :, 0:1].float() + a * bf
                    variants[f"blend{a}"] = Cb
                Ca = C.clone()
                Ca[:, :, 0:1] = bf
                variants["anchor"] = Ca

                for name, seg in variants.items():
                    rows.append({**base, "variant": name,
                                 "seam": boundary_seam(seg), "fidelity": fidelity(seg)})

                log(f"  seg{g}: seam fine={base['seam_fine']:.4f} coarse={base['seam_coarse']:.4f} "
                    f"| anchor={rows[-1]['seam']:.4f} fid_c={base['fid_coarse']:.4f}")

                boundary = F[:, :, -1:]
                del F, C, variants, ref_seg, bf
                gc.collect(); torch.cuda.empty_cache()

            manifest["records"].extend([r for r in rows if r["prompt_idx"] == pi][:6])
            del condition, noise, ref_latent, prefix, boundary
            gc.collect(); torch.cuda.empty_cache()
            log(f"  prompt {pi} done | {int(time.time() - tp)}s | elapsed {int(time.time() - t4)}s")

        variants_order = ["none"] + [f"blend{a}" for a in BLENDS] + ["anchor"]
        seams = np.array([[r["seam"] for r in rows if r["variant"] == v] for v in variants_order], dtype=np.float32)
        fids = np.array([[r["fidelity"] for r in rows if r["variant"] == v] for v in variants_order], dtype=np.float32)
        meta = {"prompt_idx": [r["prompt_idx"] for r in rows if r["variant"] == "none"],
                "segment": [r["segment"] for r in rows if r["variant"] == "none"],
                "seam_fine": [r["seam_fine"] for r in rows if r["variant"] == "none"],
                "interior_fine": [r["interior_fine"] for r in rows if r["variant"] == "none"],
                "fid_fine": [r["fid_fine"] for r in rows if r["variant"] == "none"],
                "seam_coarse": [r["seam_coarse"] for r in rows if r["variant"] == "none"],
                "interior_coarse": [r["interior_coarse"] for r in rows if r["variant"] == "none"],
                "fid_coarse": [r["fid_coarse"] for r in rows if r["variant"] == "none"]}
        np.savez_compressed(MEAS, variants=np.array(variants_order), seams=seams, fids=fids, **meta)
        manifest["variants"] = variants_order
        manifest["n_segments"] = len(meta["segment"])
        manifest["mean_seam"] = {v: float(seams[i].mean()) for i, v in enumerate(variants_order)}
        manifest["mean_fidelity"] = {v: float(fids[i].mean()) for i, v in enumerate(variants_order)}
        manifest["status"] = "complete"
        manifest["elapsed_s"] = round(time.time() - t_start, 1)
        log("mean seam: " + json.dumps(manifest["mean_seam"]))
        log("mean fidelity: " + json.dumps(manifest["mean_fidelity"]))
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
