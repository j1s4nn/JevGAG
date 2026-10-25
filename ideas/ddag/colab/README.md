# DDAG — A100 (Colab) runs

Colab has **no launch/attach API**, so A100 work needs one manual step: start an A100 Colab session
and expose SSH over a cloudflared tunnel. After that the same kernel runs unchanged.

## One-time: start the Colab A100 tunnel

1. Open <https://colab.research.google.com>, set **Runtime → Change runtime type → A100 GPU**.
2. In a cell, run:

```python
!pip -q install colab-ssh
from colab_ssh import launch_ssh_cloudflared
launch_ssh_cloudflared(password="<pick-a-password>")
```

3. The cell prints an **SSH host** (e.g. `xxxx.trycloudflare.com`) and the **port** (usually `22`).
4. Send me (or write into `~/.config/kilo/secrets/colab.env`):
   `COLAB_SSH_HOST`, `COLAB_SSH_PORT`, `COLAB_SSH_PASSWORD`.

## Then I run (no further manual steps)

```bash
# on the Colab VM
curl -sL <raw-url>/ideas/ddag/jobs/crv/crv.py -o /content/crv.py
curl -sL <raw-url>/ideas/ddag/colab/a100_crv.py -o /content/a100_crv.py
export KAGGLE_USERNAME=ajjisan KAGGLE_KEY=<from kaggle.json>
python /content/a100_crv.py
```

Outputs land in `/content/out/results/selfaware_dit/{features.npz,manifest.json}`; I download them and
re-run `ideas/ddag/scripts/crv_analysis.py` for the cross-hardware comparison.

## Why both GPUs matter

The convergence ratio (`step_last / step_first`) is a **sampler** property. Whether it transfers from
T4 to A100 is exactly the base paper's **P1** lesson: cost and behaviour are hardware-conditional. The
A100 run answers RQ6/H7 directly.

## Budget

One A100 run ≈ 15–25 min (the same kernel is ~33 min on T4). Tight watcher: 7-min polls, ≤20-min windows.
