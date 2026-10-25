# E1 break-even matrix (matched: same modes on both GPUs)

| GPU | fine | coarse | NFE f/c | break-even | measured ratio | coarse cheaper | DiT saved (s) |
|---|---|---|---|---|---|---|---|
| T4 | fine_s1 | coarse_s1 | 19.0/8.33 | 2.28 | 2.694 | no | -4.566 |
| T4 | fine_s1 | coarse_s2 | 19.0/12.67 | 1.5 | 2.423 | no | -15.47 |
| T4 | fine_s1 | coarse_s4 | 19.0/21.33 | 0.891 | 2.197 | no | -36.893 |
| T4 | fine_s2 | coarse_s1 | 34.0/8.33 | 4.08 | 3.301 | **YES** | 7.011 |
| T4 | fine_s2 | coarse_s2 | 34.0/12.67 | 2.684 | 2.969 | no | -3.893 |
| T4 | fine_s2 | coarse_s4 | 34.0/21.33 | 1.594 | 2.692 | no | -25.316 |
| T4 | fine_s4 | coarse_s1 | 64.0/8.33 | 7.68 | 3.841 | **YES** | 29.7 |
| T4 | fine_s4 | coarse_s2 | 64.0/12.67 | 5.053 | 3.454 | **YES** | 18.795 |
| T4 | fine_s4 | coarse_s4 | 64.0/21.33 | 3.0 | 3.133 | no | -2.628 |
| A100 | fine_s1 | coarse_s1 | 19.0/8.33 | 2.28 | 1.561 | **YES** | 1.237 |
| A100 | fine_s1 | coarse_s2 | 19.0/12.67 | 1.5 | 1.397 | **YES** | 0.27 |
| A100 | fine_s1 | coarse_s4 | 19.0/21.33 | 0.891 | 1.267 | no | -1.658 |
| A100 | fine_s2 | coarse_s1 | 34.0/8.33 | 4.08 | 1.95 | **YES** | 2.934 |
| A100 | fine_s2 | coarse_s2 | 34.0/12.67 | 2.684 | 1.744 | **YES** | 1.968 |
| A100 | fine_s2 | coarse_s4 | 34.0/21.33 | 1.594 | 1.583 | **YES** | 0.04 |
| A100 | fine_s4 | coarse_s1 | 64.0/8.33 | 7.68 | 2.267 | **YES** | 6.412 |
| A100 | fine_s4 | coarse_s2 | 64.0/12.67 | 5.053 | 2.028 | **YES** | 5.446 |
| A100 | fine_s4 | coarse_s4 | 64.0/21.33 | 3.0 | 1.84 | **YES** | 3.517 |

## Headline

- **T4 / coarse_s1**: cheaper than fine modes -> ['fine_s2', 'fine_s4']
- **T4 / coarse_s2**: cheaper than fine modes -> ['fine_s4']
- **T4 / coarse_s4**: cheaper than fine modes -> NONE (never cheaper)
- **A100 / coarse_s1**: cheaper than fine modes -> ['fine_s1', 'fine_s2', 'fine_s4']
- **A100 / coarse_s2**: cheaper than fine modes -> ['fine_s1', 'fine_s2', 'fine_s4']
- **A100 / coarse_s4**: cheaper than fine modes -> ['fine_s2', 'fine_s4']
