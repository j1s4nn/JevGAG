"""SelfAware-DiT capture kernel.

Runs the verified Causal-rCM causal rollout on a Kaggle GPU and captures, for every
rollout segment, the DiT's OWN internal activations at the decision point
(the I2V prefill forward on the segment boundary), together with the
coarse-unsafe label defined exactly as in the base paper
(label = 1 iff the coarse segment's reference-L1 exceeds the fine segment's).

Design (see ideas/selfaware-dit/README.md):
- fine   = c1-1_step2, 2 sampling steps/frame
- coarse = c3-3_step2, 2 sampling steps/chunk
- ref    = c1-1_step4, 4 sampling steps (the independent reference rollout)
- matched design: both modes at segment g start from the SAME boundary (the fine chain).

Outputs /kaggle/working/results/selfaware_dit/:
  features.npz   pooled per-block activations + labels + groups
  manifest.json  protocol + per-segment records + timing

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
FPS = 8
MID = [15 / 16, 5 / 6, 5 / 8]
N_REF_STEPS = 4
N_FINE_STEPS = 2
N_COARSE_STEPS = 2
DECODE_PIXELS = False  # SA_DECODE  (T4 VAE decode dominates wall-clock and OOMs when batched)
BLOCK_RE = re.compile(r"blocks\.\d+$")
RCM_DIR = Path("/tmp/rcm")

OUT_DIR = Path(os.environ.get("SA_OUT_ROOT", "/kaggle/working")) / "results" / "selfaware_dit"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FEATURES = OUT_DIR / "features.npz"
MANIFEST = OUT_DIR / "manifest.json"

PROMPTS = [
    {"stratum": "object_class", "prompt": "a boat"},
    {"stratum": "object_class", "prompt": "a book"},
    {"stratum": "object_class", "prompt": "a bowl of soup"},
    {"stratum": "object_class", "prompt": "a red sports car"},
    {"stratum": "object_class", "prompt": "a white swan on a lake"},
    {"stratum": "scene", "prompt": "baseball stadium"},
    {"stratum": "scene", "prompt": "basement"},
    {"stratum": "scene", "prompt": "a busy city street at night"},
    {"stratum": "scene", "prompt": "a quiet library interior"},
    {"stratum": "scene", "prompt": "a sandy beach at sunset"},
    {"stratum": "motion", "prompt": "a bird flying over a snowy forest"},
    {"stratum": "motion", "prompt": "a bird soaring gracefully in the sky"},
    {"stratum": "motion", "prompt": "a waterfall crashing into a river"},
    {"stratum": "motion", "prompt": "a train speeding through the countryside"},
    {"stratum": "motion", "prompt": "a campfire burning at night"},
    {"stratum": "human_action", "prompt": "A person is bungee jumping"},
    {"stratum": "human_action", "prompt": "A person is canoeing or kayaking"},
    {"stratum": "human_action", "prompt": "A person is riding a bicycle down a hill"},
    {"stratum": "human_action", "prompt": "A person is dancing in the rain"},
    {"stratum": "human_action", "prompt": "A person is skiing down a snowy slope"},
    {"stratum": "human_action", "prompt": "A person is swimming in the ocean"},
    {"stratum": "object_class", "prompt": "a wooden chair in a room"},
    {"stratum": "scene", "prompt": "an empty highway at dawn"},
    {"stratum": "motion", "prompt": "leaves falling from a tree"},
]


def log(m):
    print(m, flush=True)


def find_under(root, filename):
    root = Path(root)
    return next((p for p in root.rglob(filename) if p.is_file()), None) if root.exists() else None


def install_deps():
    pkgs = [
        "einops", "transformers", "sentencepiece", "ftfy", "regex", "imageio",
        "imageio-ffmpeg", "tqdm", "fvcore", "omegaconf", "attrs", "pyyaml", "iopath",
        "termcolor", "pynvml", "pandas", "loguru", "safetensors", "cloudpickle", "dill",
        "matplotlib", "packaging", "opencv-python-headless",
    ]
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


def to_frames(video):
    px = (1 + video.float().cpu().clamp(-1, 1)) / 2
    if px.ndim == 5:
        return (px.permute(0, 2, 3, 4, 1).numpy() * 255).astype(np.uint8)
    return (px.permute(1, 2, 3, 0).numpy() * 255).astype(np.uint8)


def pool_tokens(t):
    t = t.detach().float()
    if t.ndim == 3:
        return t.mean(dim=1)[0].cpu()
    return t.reshape(t.shape[0], -1).mean(dim=1)[0].cpu()


class Capture:
    """Records, per net forward call, the pooled output of every DiT block."""

    def __init__(self, net):
        self.net = net
        self.calls = []
        self._cur = None
        self._h = []
        self._h.append(net.register_forward_pre_hook(self._pre))
        for n, m in net.named_modules():
            if BLOCK_RE.fullmatch(n):
                self._h.append(m.register_forward_hook(self._blk(n)))
        self._h.append(net.register_forward_hook(self._post))

    def _pre(self, module, args):
        self._cur = {}
        self.calls.append(self._cur)

    def _blk(self, name):
        def hook(module, args, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(t) and self._cur is not None:
                self._cur[name] = pool_tokens(t)
        return hook

    def _post(self, module, args, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if torch.is_tensor(t) and self._cur is not None:
            tf = t.detach().float()
            self._cur["__vnorm__"] = torch.tensor([float(tf.norm()), float(tf.abs().mean())])

    def close(self):
        for h in self._h:
            h.remove()
        self._h = []


def stack_call(call, names):
    present = [n for n in names if n in call]
    if not present:
        return [], np.zeros((0, 0), dtype=np.float16)
    return present, np.stack([call[n].numpy().astype(np.float16) for n in present], axis=0)


def seg_l1(seg_frames, ref_frames, seg_range):
    ref = ref_frames[seg_range].astype(np.float32) / 255.0
    k = min(seg_frames.shape[0], ref.shape[0])
    a = seg_frames[:k].astype(np.float32) / 255.0
    return float(np.mean(np.abs(a - ref[:k]))), k


def main():
    t_start = time.time()
    T = {}
    manifest = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "kernel": "selfaware-dit capture",
        "protocol": {
            "seed": SEED, "horizon": HORIZON, "limit": LIMIT,
            "n_ref_steps": N_REF_STEPS, "n_fine_steps": N_FINE_STEPS, "n_coarse_steps": N_COARSE_STEPS,
            "fine_ckpt": "c1-1_step2", "coarse_ckpt": "c3-3_step2", "ref_ckpt": "c1-1_step4",
            "label": "coarse_unsafe = 1[ refL1_coarse > refL1_fine ]",
            "label_space": "pixel" if DECODE_PIXELS else "latent",
            "labels": ["labels(primary)", "labels_latent", "labels_smooth(jitter)"],
        },
        "records": [],
        "status": "running",
    }
    try:
        t = time.time()
        install_deps()
        T["deps_s"] = round(time.time() - t, 1)
        gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "none"
        manifest["gpu"] = gpu
        log(f"gpu={gpu} deps={T['deps_s']}s")
        if not torch.cuda.is_available():
            raise RuntimeError("no GPU")

        if RCM_DIR.exists():
            shutil.rmtree(RCM_DIR)
        t = time.time()
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/NVlabs/rcm.git", str(RCM_DIR)], check=True, capture_output=True)
        T["clone_s"] = round(time.time() - t, 1)
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
        manifest["dataset_paths"] = {k: str(v) for k, v in paths.items()}
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
        manifest["latent"] = {"frames": int(latent_frames), "h": int(latent_h), "w": int(latent_w), "c": int(latent_c)}

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
            for prompt in prompts:
                ids, mask = tok(prompt, return_mask=True, add_special_tokens=True)
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
            frames = to_frames(video)
            del video
            torch.cuda.empty_cache()
            return frames

        def decode_batch(latents):
            b = torch.cat([l.float() for l in latents], dim=0).cuda()
            video = tokenizer.decode(b)
            out = to_frames(video)
            del b, video
            torch.cuda.empty_cache()
            if out.ndim == 4:
                out = out[None]
            return [out[i] for i in range(out.shape[0])]

        names = None
        feat_prefill, feat_gen, feat_meta = [], [], []
        labels, labels_latent, labels_smooth = [], [], []
        N = min(LIMIT, len(PROMPTS))
        log(f"encoding {N} prompts (T5 loaded once)...")
        t = time.time()
        EMBS = encode_prompts([PROMPTS[i]["prompt"] for i in range(N)])
        T["encode_s"] = round(time.time() - t, 1)
        log(f"encoded in {T['encode_s']}s")
        t4 = time.time()

        for pi in range(N):
            spec = PROMPTS[pi]
            prompt = spec["prompt"]
            tp = time.time()
            log(f"=== prompt {pi}: {prompt}")
            condition = {"crossattn_emb": EMBS[pi].to(**TENSOR_KWARGS)}
            gen0 = torch.Generator(device="cuda").manual_seed(SEED)
            noise = torch.randn(1, latent_c, latent_frames, latent_h, latent_w, dtype=torch.float32, device="cuda", generator=gen0).cpu()

            # --- reference rollout (c1-1 s4) ---
            net = get_net("reference")
            if names is None:
                names = [n for n, _ in net.named_modules() if BLOCK_RE.fullmatch(n)]
                manifest["block_names"] = names
                manifest["n_blocks"] = len(names)
            steps = build_few_step_t_steps(N_REF_STEPS, 1600, MID, "cuda")
            _, blocks, _ = make_block_pattern(latent_frames, latent_h, latent_w, 1, 1, net.get_spatial_patch_size())
            counts = expand_steps_per_chunk(None, blocks, N_REF_STEPS)
            sched = build_few_step_schedules_per_chunk(counts, None, 1600, MID, [], "cuda")
            gen = torch.Generator(device="cuda").manual_seed(SEED + 1000)
            net.cuda()
            ref_latent = causal_rollout_sampling(net, noise.cuda(), steps, sched, None, condition, None, 1.0, 1, 1, False, generator=gen).detach().cpu()
            net.cpu()
            torch.cuda.empty_cache()
            ref_frames = None
            if DECODE_PIXELS:
                t = time.time()
                ref_frames = decode(ref_latent)
                T["ref_decode_s"] = round(T.get("ref_decode_s", 0.0) + time.time() - t, 1)
            del sched, counts
            gc.collect()
            torch.cuda.empty_cache()

            # --- shared coarse prefix (3 latent frames) ---
            net = get_net("coarse")
            steps = build_few_step_t_steps(N_COARSE_STEPS, 1600, MID, "cuda")
            _, blocks, _ = make_block_pattern(3, latent_h, latent_w, 3, 3, net.get_spatial_patch_size())
            counts = expand_steps_per_chunk(None, blocks, N_COARSE_STEPS)
            sched = build_few_step_schedules_per_chunk(counts, None, 1600, MID, [], "cuda")
            gen = torch.Generator(device="cuda").manual_seed(SEED + 2000)
            net.cuda()
            prefix = causal_rollout_sampling(net, noise[:, :, :3].cuda(), steps, sched, None, condition, None, 1.0, 3, 3, False, generator=gen).detach().cpu()
            net.cpu()
            torch.cuda.empty_cache()
            del sched, counts
            gc.collect()
            torch.cuda.empty_cache()

            boundary = prefix[:, :, -1:]
            seg_data = []

            for g in range(1, 5):
                def run_segment(mode):
                    net = get_net(mode)
                    nsteps = N_COARSE_STEPS if mode == "coarse" else N_FINE_STEPS
                    chunk = 3 if mode == "coarse" else 1
                    spc = None if mode == "coarse" else [N_FINE_STEPS]
                    cap = Capture(net)
                    steps = build_few_step_t_steps(nsteps, 1600, MID, "cuda")
                    _, blocks, _ = make_block_pattern(4, latent_h, latent_w, 1, chunk, net.get_spatial_patch_size())
                    counts = expand_steps_per_chunk(spc, blocks - 1, nsteps)
                    sched = build_few_step_schedules_per_chunk(counts, spc, 1600, MID, [], "cuda")
                    nz = noise[:, :, 3 * g:3 * (g + 1)].cuda()
                    gen = torch.Generator(device="cuda").manual_seed(SEED + 2000 + g)
                    net.cuda()
                    latent = causal_i2v_rollout_sampling(net, boundary.cuda(), nz, steps, sched, spc, condition, None, 1.0, chunk, False, generator=gen)
                    fresh = latent[:, :, 1:].detach().cpu()
                    calls = cap.calls
                    cap.close()
                    net.cpu()
                    torch.cuda.empty_cache()
                    del latent, nz, sched, counts
                    gc.collect()
                    torch.cuda.empty_cache()
                    return fresh, calls

                fine_lat, fine_calls = run_segment("fine")
                coarse_lat, coarse_calls = run_segment("coarse")
                ref_seg_lat = ref_latent[:, :, 3 + 3 * (g - 1):3 + 3 * g]
                ll1_fine = float((fine_lat.float() - ref_seg_lat.float()).abs().mean())
                ll1_coarse = float((coarse_lat.float() - ref_seg_lat.float()).abs().mean())
                lj_fine = float((fine_lat[:, :, 1:] - fine_lat[:, :, :-1]).abs().mean())
                lj_coarse = float((coarse_lat[:, :, 1:] - coarse_lat[:, :, :-1]).abs().mean())
                # post-generation (free) evidence for the closed loop
                bf = boundary.float()
                seam_coarse = float((coarse_lat[:, :, 0:1].float() - bf).abs().mean())
                seam_fine = float((fine_lat[:, :, 0:1].float() - bf).abs().mean())
                disc = float((coarse_lat.float() - fine_lat.float()).abs().mean())
                ic = float((coarse_lat[:, :, 1:].float() - coarse_lat[:, :, :-1].float()).abs().mean())
                ifn = float((fine_lat[:, :, 1:].float() - fine_lat[:, :, :-1].float()).abs().mean())
                vn_fine = [[float(c["__vnorm__"][0]), float(c["__vnorm__"][1])] if "__vnorm__" in c else [0.0, 0.0] for c in fine_calls]
                vn_coarse = [[float(c["__vnorm__"][0]), float(c["__vnorm__"][1])] if "__vnorm__" in c else [0.0, 0.0] for c in coarse_calls]
                seg_data.append({"g": g, "fine_lat": fine_lat, "coarse_lat": coarse_lat, "fine_calls": fine_calls,
                                 "ll1_fine": ll1_fine, "ll1_coarse": ll1_coarse, "lj_fine": lj_fine, "lj_coarse": lj_coarse,
                                 "seam_coarse": seam_coarse, "seam_fine": seam_fine, "disc": disc,
                                 "interior_coarse": ic, "interior_fine": ifn,
                                 "vn_fine": vn_fine, "vn_coarse": vn_coarse,
                                 "n_fine_calls": len(fine_calls), "n_coarse_calls": len(coarse_calls)})
                boundary = fine_lat[:, :, -1:]

            frames_list = None
            if DECODE_PIXELS:
                t = time.time()
                frames_list = [decode(s["fine_lat"]) for s in seg_data] + [decode(s["coarse_lat"]) for s in seg_data]
                T["seg_decode_s"] = round(T.get("seg_decode_s", 0.0) + time.time() - t, 1)

            for i, s in enumerate(seg_data):
                g = s["g"]
                yl = int(s["ll1_coarse"] > s["ll1_fine"])
                yls = int(s["lj_coarse"] > s["lj_fine"])
                if DECODE_PIXELS:
                    seg_range = slice(4 * (3 + 3 * (g - 1)), 4 * (3 + 3 * g))
                    l1_fine, k = seg_l1(frames_list[i], ref_frames, seg_range)
                    l1_coarse, _ = seg_l1(frames_list[len(seg_data) + i], ref_frames, seg_range)

                    def jitter(fr):
                        a = fr[:k].astype(np.float32) / 255.0
                        return float(np.mean(np.abs(np.diff(a, axis=0)))) if k > 1 else 0.0
                    j_fine, j_coarse = jitter(frames_list[i]), jitter(frames_list[len(seg_data) + i])
                    y, ys = int(l1_coarse > l1_fine), int(j_coarse > j_fine)
                else:
                    l1_fine, l1_coarse, j_fine, j_coarse = s["ll1_fine"], s["ll1_coarse"], s["lj_fine"], s["lj_coarse"]
                    y, ys = yl, yls

                fine_calls = s["fine_calls"]
                _, pre_mat = stack_call(fine_calls[0], names)
                gen_mats = [stack_call(c, names)[1] for c in fine_calls[1:] if c]
                gen_mat = np.mean(np.stack(gen_mats, axis=0), axis=0).astype(np.float16) if gen_mats else np.zeros_like(pre_mat)
                vn = fine_calls[0].get("__vnorm__", torch.tensor([0.0, 0.0]))

                feat_prefill.append(pre_mat)
                feat_gen.append(gen_mat)
                labels.append(y)
                labels_latent.append(yl)
                labels_smooth.append(ys)
                feat_meta.append({
                    "prompt_idx": pi, "stratum": spec["stratum"], "prompt": prompt, "segment": g,
                    "l1_fine": l1_fine, "l1_coarse": l1_coarse, "label_coarse_unsafe": y,
                    "ll1_fine": s["ll1_fine"], "ll1_coarse": s["ll1_coarse"], "label_latent": yl,
                    "jitter_fine": j_fine, "jitter_coarse": j_coarse, "label_smooth": ys,
                    "lj_fine": s["lj_fine"], "lj_coarse": s["lj_coarse"], "label_latent_smooth": yls,
                    "disc_coarse_fine": s["disc"], "seam_coarse": s["seam_coarse"], "seam_fine": s["seam_fine"],
                    "interior_coarse": s["interior_coarse"], "interior_fine": s["interior_fine"],
                    "vn_fine": s["vn_fine"], "vn_coarse": s["vn_coarse"],
                    "n_fine_calls": s["n_fine_calls"], "n_coarse_calls": s["n_coarse_calls"],
                    "prefill_vnorm": float(vn[0]), "prefill_vabsmean": float(vn[1]),
                })
                log(f"  seg{g}: l1f={l1_fine:.4f} l1c={l1_coarse:.4f} y={y} | ll1f={s['ll1_fine']:.4f} ll1c={s['ll1_coarse']:.4f} yl={yl}")

            del seg_data, frames_list, ref_frames, prefix, boundary
            gc.collect()
            torch.cuda.empty_cache()
            manifest["records"].extend(feat_meta[-4:])
            log(f"  prompt {pi} done | {int(time.time() - tp)}s | elapsed {int(time.time() - t4)}s")

        pre = np.stack(feat_prefill, axis=0) if feat_prefill else np.zeros((0, 0, 0), dtype=np.float16)
        genf = np.stack(feat_gen, axis=0) if feat_gen else np.zeros((0, 0, 0), dtype=np.float16)

        def pad_vn(key):
            mx = max((len(r[key]) for r in feat_meta), default=0)
            arr = np.zeros((len(feat_meta), mx, 2), dtype=np.float32)
            for i, r in enumerate(feat_meta):
                v = r[key]
                if v:
                    arr[i, :len(v)] = np.asarray(v, dtype=np.float32)
            return arr

        np.savez_compressed(
            FEATURES, prefill=pre, gen=genf,
            labels=np.asarray(labels, dtype=np.int64),
            labels_latent=np.asarray(labels_latent, dtype=np.int64),
            labels_smooth=np.asarray(labels_smooth, dtype=np.int64),
            prompt_idx=np.asarray([r["prompt_idx"] for r in feat_meta], dtype=np.int64),
            segment=np.asarray([r["segment"] for r in feat_meta], dtype=np.int64),
            disc=np.asarray([r["disc_coarse_fine"] for r in feat_meta], dtype=np.float32),
            seam_coarse=np.asarray([r["seam_coarse"] for r in feat_meta], dtype=np.float32),
            seam_fine=np.asarray([r["seam_fine"] for r in feat_meta], dtype=np.float32),
            interior_coarse=np.asarray([r["interior_coarse"] for r in feat_meta], dtype=np.float32),
            interior_fine=np.asarray([r["interior_fine"] for r in feat_meta], dtype=np.float32),
            ll1_fine=np.asarray([r["ll1_fine"] for r in feat_meta], dtype=np.float32),
            ll1_coarse=np.asarray([r["ll1_coarse"] for r in feat_meta], dtype=np.float32),
            vn_coarse=pad_vn("vn_coarse"),
            vn_fine=pad_vn("vn_fine"),
        )
        manifest["n_segments"] = int(len(labels))
        manifest["n_positive"] = int(sum(labels))
        manifest["n_positive_latent"] = int(sum(labels_latent))
        manifest["n_positive_smooth"] = int(sum(labels_smooth))
        manifest["feature_shape"] = list(pre.shape)
        manifest["timing"] = T
        manifest["status"] = "complete"
        manifest["elapsed_s"] = round(time.time() - t_start, 1)
        log(f"saved {FEATURES} shape={pre.shape} pos(pixel)={sum(labels)} pos(latent)={sum(labels_latent)}/{len(labels)}")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["traceback"] = traceback.format_exc()
        manifest["timing"] = T
        log(traceback.format_exc())
    finally:
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        log(f"manifest: {MANIFEST}")


if __name__ == "__main__":
    main()

