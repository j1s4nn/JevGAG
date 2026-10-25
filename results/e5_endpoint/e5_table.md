# E5 — image-Jev endpoint vs CLIP subject consistency

| policy | Jev consistency | Jev same-subject | CLIP subj.cons. |
|---|---|---|---|
| adaptive_warp | 2.081 | 0.410 | 0.980 |
| alternating50 | 1.626 | 0.268 | 0.959 |
| fixed_coarse | 1.775 | 0.230 | 0.949 |
| fixed_fine | 2.151 | 0.416 | 0.984 |
| random50 | 1.978 | 0.286 | 0.965 |
| reference | 3.461 | 0.894 | 0.990 |

Spearman(Jev consistency, CLIP) = **+0.943**
Spearman(Jev same-subject, CLIP) = **+1.000**
