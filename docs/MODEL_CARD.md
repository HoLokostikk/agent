# Model card: ft-v1 (template, fill in after the GPU run)

| field | value |
|---|---|
| base model | Qwen/Qwen3.8-27B @ `<gpu/PINNED_REVISION>` |
| method | QLoRA (NF4, double quant), r=`<r>`, alpha=`<alpha>`, all attention + MLP projections |
| data | `docs/DATASET_CARD.md`, `<n>` trajectories, sha256 `<from run_manifest.json>` |
| loss | assistant tokens only (`src/agent/training/masking.py`) |
| training | `<epochs>` epochs, lr `<lr>`, `<GPU>`, `<hours>` h, manifest `training/runs/<run>/run_manifest.json` |
| licence | Apache-2.0 (same as the base model) |

## Intended use

Inside this repository's harness (its tool set and prompt `system.v1`). Gains,
if any, are expected in tool-call validity and steps per task, not in general
coding ability.

## Evaluation (same harness, same prompt, same tasks)

| metric | base | ft-v1 | difference [95% CI] | report |
|---|---|---|---|---|
| pass@1, golden set | | | | |
| pass^3 | | | | |
| invalid tool-call rate | | | | |
| mean turns per task | | | | |
| tool-call eval accuracy | | | | |

## Limitations

- Trained on mutation bugs from a handful of Python repositories; real issues are harder.
- Small eval set: intervals are wide; a difference whose CI includes zero is not a result.
