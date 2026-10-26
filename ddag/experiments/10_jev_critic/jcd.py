"""DDAG E7 — JCD: Jev as a typed critic for granularity safety (T4 / A100).

For each prompt and segment we generate the matched coarse and fine segment from a shared
boundary, decode the coarse segment to pixels, and ask Jev (image, typed questions) about it.
Jev sees ONLY pixels -- no measured statistics -- so this is a fair test of whether a decision
model can independently certify granularity safety.

Questions (Noul):
  q_discontinuity : "Is there a visible temporal discontinuity or artifact ..."
  q_plausible     : "Is the motion physically plausible and temporally coherent ..."

Outputs <SA_OUT_ROOT>/results/jcd/: measurements.npz + manifest.json

LIMIT = 16  # SA_LIMIT
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import base64
import gc
import io
import json
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
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
JEV_MODEL = "jev-1.14.0"
JEV_URL = "https://api.typesafe.ai/v1/systemone"

OUT_DIR = Path(os.environ.get("SA_OUT_ROOT", "/kaggle/working")) / "results" / "jcd"
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
]


def log(m):
    print(m, flush=True)


def jev_key():
    env = os.environ.get("TYPESAFE_API_KEY")
    if env:
        return env
    cands = []
    p = os.environ.get("TYPESAFE_ENV")
    if p:
        cands.append(p)
    cands += ["/content/typesafe.env"]
    root = Path(os.environ.get("SA_INPUT_ROOT", "/kaggle/input"))
    if root.exists():
        cands += [str(x) for x in root.rglob("typesafe.env")]
        cands += [str(x) for x in root.rglob("*.env")]
    for cand in cands:
        if cand and os.path.exists(cand):
            for line in open(cand, encoding="utf-8"):
                line = line.strip()
                if line.startswith("TYPESAFE_API_KEY"):
                    return line.split("=", 1)[1].strip().strip('"')
    return None


def jev_ask(key, state, questions, image_png=None, ref="frame-ref-001", part="frame"):
    """TypeSafe requires multipart/form-data for images; assets is a map keyed by a
    reference string that must appear somewhere in state/instructions/criteria."""
    boundary = "----jevboundary7f3a"
    body = {"model": JEV_MODEL, "questions": questions}
    if image_png is not None:
        body["state"] = state
        body["assets"] = {ref: {"type": "image", "part": part}}
    else:
        body["state"] = state
    chunks = []
    chunks.append(f"--{boundary}\r\n".encode())
    chunks.append(b'Content-Disposition: form-data; name="request"\r\n')
    chunks.append(b"Content-Type: application/json\r\n\r\n")
    chunks.append(json.dumps(body).encode() + b"\r\n")
    if image_png is not None:
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(f'Content-Disposition: form-data; name="{part}"; filename="{part}.png"\r\n'.encode())
        chunks.append(b"Content-Type: image/png\r\n\r\n")
        chunks.append(image_png + b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    data = b"".join(chunks)
    headers = {"Authorization": f"Bearer {key}",
               "Content-Type": f"multipart/form-data; boundary={boundary}"}
    if image_png is not None:
        headers["TypeSafe-Danger-Zone-Images"] = "true"
    req = urllib.request.Request(JEV_URL, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return {"_error": f"{e.code}: {e.read().decode()[:300]}"}
    except Exception as e:
        return {"_error": str(e)}


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


def png_bytes(frames_u8, idxs):
    """Compose the selected frames side by side into one PNG within Jev's limits
    (each side 280..1568 px, <= 2,458,624 px total)."""
    from PIL import Image
    ims = [Image.fromarray(frames_u8[i]) for i in idxs if i < frames_u8.shape[0]]
    if not ims:
        return None
    w = sum(im.width for im in ims)
    canvas = Image.new("RGB", (w, max(im.height for im in ims)))
    x = 0
    for im in ims:
        canvas.paste(im, (x, 0))
        x += im.width
    canvas.thumbnail((1536, 1536))
    if min(canvas.size) < 280:
        scale = 280 / min(canvas.size)
        canvas = canvas.resize((max(280, int(canvas.width * scale)), max(280, int(canvas.height * scale))))
    buf = io.BytesIO()
    canvas.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def main():
    t_start = time.time()
    manifest = {"run_id": datetime.now().strftime("%Y%m%d_%H%M%S"), "kernel": "ddag-e7-jcd",
                "protocol": {"seed": SEED, "horizon": HORIZON, "limit": LIMIT, "jev_model": JEV_MODEL},
                "status": "running", "records": []}
    try:
        install_deps()
        manifest["gpu"] = torch.cuda.get_device_name() if torch.cuda.is_available() else "none"
        key = jev_key()
        manifest["jev_key_present"] = bool(key)
        log(f"gpu={manifest['gpu']} jev_key={bool(key)}")
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

        def get_net(key_):
            if key_ not in NETS:
                with init_weights_on_device():
                    net = instantiate(DIT_CONFIGS["1.3B"]).eval()
                load_dit_weights(net, str(paths[key_]))
                net.to(**TENSOR_KWARGS).cpu()
                torch.cuda.empty_cache()
                NETS[key_] = net
            return NETS[key_]

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

        rows = []
        t4 = time.time()
        for pi in range(N):
            prompt = PROMPTS[pi]
            tp = time.time()
            log(f"=== prompt {pi}: {prompt}")
            condition = {"crossattn_emb": EMBS[pi].to(**TENSOR_KWARGS)}
            gen0 = torch.Generator(device="cuda").manual_seed(SEED)
            noise = torch.randn(1, latent_c, latent_frames, latent_h, latent_w, dtype=torch.float32, device="cuda", generator=gen0).cpu()

            net = get_net("reference")
            steps = build_few_step_t_steps(N_REF_STEPS, 1600, MID, "cuda")
            _, blocks, _ = make_block_pattern(latent_frames, latent_h, latent_w, 1, 1, net.get_spatial_patch_size())
            counts = expand_steps_per_chunk(None, blocks, N_REF_STEPS)
            sched = build_few_step_schedules_per_chunk(counts, None, 1600, MID, [], "cuda")
            gen = torch.Generator(device="cuda").manual_seed(SEED + 1000)
            net.cuda()
            ref_latent = causal_rollout_sampling(net, noise.cuda(), steps, sched, None, condition, None, 1.0, 1, 1, False, generator=gen).detach().cpu()
            net.cpu(); torch.cuda.empty_cache(); del sched, counts; gc.collect(); torch.cuda.empty_cache()

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
                C = run_segment("coarse")
                F = run_segment("fine")
                rng = slice(3 + 3 * (g - 1), 3 + 3 * g)
                ll1_c = float((C.float() - ref_latent[:, :, rng].float()).abs().mean())
                ll1_f = float((F.float() - ref_latent[:, :, rng].float()).abs().mean())
                y = int(ll1_c > ll1_f)

                coarse_frames = decode(C)
                png = png_bytes(coarse_frames, [0, coarse_frames.shape[0] - 1])
                state = {"prompt": prompt, "frames": "frame-ref-001"}
                q = {
                    "discontinuity": {"type": "noul", "instructions": "True if the frames in `frames` show a visible temporal discontinuity, jump, or artifact a viewer would notice."},
                    "plausible": {"type": "noul", "instructions": "True if the motion and content in `frames` are physically plausible and temporally coherent."},
                }
                ans = jev_ask(key, state, q, image_png=png)
                a = ans.get("answers", {}) if isinstance(ans, dict) else {}
                rows.append({
                    "prompt_idx": pi, "prompt": prompt, "segment": g,
                    "ll1_coarse": ll1_c, "ll1_fine": ll1_f, "label_coarse_unsafe": y,
                    "jev_discontinuity": (a.get("discontinuity") or {}).get("noul"),
                    "jev_plausible": (a.get("plausible") or {}).get("noul"),
                    "jev_error": ans.get("_error") if isinstance(ans, dict) else None,
                })
                log(f"  seg{g}: y={y} jev_disc={rows[-1]['jev_discontinuity']} jev_plaus={rows[-1]['jev_plausible']} err={rows[-1]['jev_error']}")

                boundary = F[:, :, -1:]
                del C, F, coarse_frames
                gc.collect(); torch.cuda.empty_cache()

            del condition, noise, ref_latent, prefix, boundary
            gc.collect(); torch.cuda.empty_cache()
            manifest["records"].extend([r for r in rows if r["prompt_idx"] == pi])
            log(f"  prompt {pi} done | {int(time.time() - tp)}s | elapsed {int(time.time() - t4)}s")

        def col(name):
            return np.asarray([r[name] if r[name] is not None else np.nan for r in rows], dtype=np.float32)

        np.savez_compressed(
            MEAS,
            labels=np.asarray([r["label_coarse_unsafe"] for r in rows], dtype=np.int64),
            ll1_coarse=np.asarray([r["ll1_coarse"] for r in rows], dtype=np.float32),
            ll1_fine=np.asarray([r["ll1_fine"] for r in rows], dtype=np.float32),
            prompt_idx=np.asarray([r["prompt_idx"] for r in rows], dtype=np.int64),
            segment=np.asarray([r["segment"] for r in rows], dtype=np.int64),
            jev_discontinuity=col("jev_discontinuity"),
            jev_plausible=col("jev_plausible"),
        )
        manifest["n_segments"] = len(rows)
        manifest["n_positive"] = int(sum(r["label_coarse_unsafe"] for r in rows))
        manifest["n_jev_ok"] = int(sum(1 for r in rows if r["jev_error"] is None and r["jev_discontinuity"] is not None))
        manifest["status"] = "complete"
        manifest["elapsed_s"] = round(time.time() - t_start, 1)
        log(f"saved {MEAS} segments={len(rows)} jev_ok={manifest['n_jev_ok']}")
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
