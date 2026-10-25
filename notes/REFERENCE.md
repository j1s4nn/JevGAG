# REFERENCE.md — read-only reuse map (prior project → new experiments)

Old root (do not modify):
`C:\Users\jisan\OneDrive\Desktop\research\idea_05_adaptive_granularity_WM\04_Experiment`

| New E | Old artifact (read-only) |
|---|---|
| E1 break-even | `results/f1_mode_cost/`, `results/a100_mode_cost/`, `results/jev_p100_timing/`, `results/a100_timing/` |
| E2 signal | `results/phase5b_heldout_00..07/` (288 segments), `results/jev_signal/` (text-Jev baseline), `results/f4lite/videos/`, `results/rq4_monotonicity/videos/` |
| E3/E6 online | `results/f1_mode_cost/`, `results/f1_mode_cost_noisy/`, `results/a100_online_v2/`, `results/a100_online_jev/`, `results/a100_online_jev_phase2/`, `results/a100_online_jev_phase3/` |
| E4 controller | `results/rq3_controller/`, `results/a100_rq3/`, `config/rq3_threshold.json` |
| E5 endpoint | all 430 videos: `a100_online_jev*`, `a100_rq3`, `f4lite`, `rq4`, `jev_hold_mode`, `f1_mode_cost` |
| figures | `paper_materials/figures/`, `paper_materials/qualitative_frames/`, `paper_materials/image_contact_sheets/` |
| media | Kaggle `ajjisan/jev-adaptive-granularity-figures`, `ajjisan/jev-adaptive-granularity-videos`, `ajjisan/f4lite-videos` |
| checkpoints | Kaggle `ajjisan/causal-rcm-ckpts`, `ajjisan/wan21-t5`, `ajjisan/wan21-dit-vae` |

Rule: never edit the old root. New code reads it; new results are written only under this folder's `results/`.
