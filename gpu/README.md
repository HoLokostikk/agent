# Measuring everything on a rented GPU

Everything that needs the 27B model is scripted. The split is deliberate:

- **the pod** (rented GPU) serves the model with vLLM and trains the adapter;
- **the laptop** runs the agent, its Docker sandbox and every eval, talking to
  the pod through an SSH tunnel.

GPU pods (RunPod, Vast.ai) are containers without Docker, and the sandbox
needs Docker. Keeping the harness on the laptop also means the base model and
the fine-tune are measured by exactly the same code on the same machine.

> None of these scripts has run on a GPU yet. Expect to fix a flag or two
> (vLLM renames options between releases); write down what you changed in
> `EXPERIMENTS.md`.

## What to rent

| step | VRAM needed | suggestion |
|---|---|---|
| serve Qwen3.8-27B in BF16 (~54 GB weights) | 80 GB | 1x H100 80GB or A100 80GB |
| serve in FP8 (`QUANTIZATION=fp8`) | 48 GB | 1x L40S / A6000 48GB (FP8 needs Ada/Hopper) |
| QLoRA training, 16K-token trajectories | 80 GB | same pod |

Pick a pod with SSH access and at least 150 GB of disk. Set a spending limit in
the provider's dashboard before starting. Check current prices before renting;
the budget below assumes about $2 per hour.

## Order of work and rough cost

| # | where | command | rough time |
|---|---|---|---|
| 1 | pod | `git clone https://github.com/HoLokostikk/agent /workspace/agent && cd /workspace/agent && bash gpu/pod_setup.sh` | 20–40 min (download) |
| 2 | pod | `bash gpu/pod_serve.sh` (keep it running, e.g. in tmux) | — |
| 3 | laptop | `ssh -N -L 8000:127.0.0.1:8000 -p <port> root@<ip> &` then `export POD_SSH="-p <port> root@<ip>"` | — |
| 4 | laptop | `bash gpu/laptop_measure.sh baseline` | ~1 h |
| 5 | laptop | `bash gpu/laptop_measure.sh bench`; restart vLLM with `PREFIX_CACHING=0`; `bash gpu/laptop_measure.sh bench-nocache` | ~20 min |
| 6 | laptop | `PER_REPO=50 bash gpu/laptop_measure.sh tasks` (CPU only: do it before renting) | 30–60 min |
| 7 | laptop | `bash gpu/laptop_measure.sh collect` | 3–4 h for ~200 tasks × 2 |
| 8 | laptop → pod | `scp -P <port> -r training/data/out root@<ip>:/workspace/agent/training/data/` | — |
| 9 | pod | stop vLLM; `bash gpu/pod_train.sh` (prints the loss masking first: read it) | 2–4 h |
| 10 | pod | `LORA="ft-v1=training/runs/qwen27b-r16/adapter" bash gpu/pod_serve.sh` | — |
| 11 | laptop | `bash gpu/laptop_measure.sh sft-eval`, then `bash gpu/laptop_measure.sh compare` | ~1 h |
| 12 | both | copy `evals/results/`, `serving/results/`, `training/runs/*/run_manifest.json` and `gpu/*.txt` into the repo and commit | — |

Total: roughly 8–11 GPU hours, about $20–30 at $2 per hour. Step 6 needs no
GPU. Ablations (`LORA_R=64`, `FRACTION=0.25`) add 2–3 hours each.

## Cost per task when self-hosting

Self-hosted aliases are priced at $0 per token in the registry; the real cost
is GPU time. From any results file:

    cost per task            = mean wall time per run / concurrency × $/hour / 3600
    cost per successful task = cost per task / pass@1

Write the hourly rate you paid next to the numbers in `EXPERIMENTS.md`.

## Teacher model

Step 7 uses the served base model as its own teacher: rejection sampling keeps
only trajectories whose tests pass (self-distillation; Apache-2.0, no terms of
service issue). A larger open-weight teacher whose licence allows training on
its outputs would likely give better data; record the choice and its licence
in `docs/DATASET_CARD.md`.

## Reproducibility checklist

- `gpu/PINNED_REVISION` holds the exact model commit; `configs/models.yaml`
  must match it (the laptop script checks, and evals refuse unpinned models).
- `gpu/pod_freeze.txt` and `gpu/pod_gpu.txt` record the software and hardware.
- Each training run writes `run_manifest.json` (git sha, data hash, config,
  pip freeze).
- Each results file records git revision, model revision, prompt version and
  sampling.
