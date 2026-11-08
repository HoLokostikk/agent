# Dataset card: agent trajectories (template, fill in after collection)

| field | value |
|---|---|
| tasks | AST-mutation bug-fix tasks (`agent data mutate`) from: `<repos and commits>` |
| teacher | `<alias, model, revision>`; licence `<licence>`; provider terms `<link / self-hosted>` |
| harness | this repo @ `<git sha>`, prompt `system.v1` |
| selection | tests pass, stop reason `completed`, no invalid tool calls, length limits, no secrets/PII, not an eval task, < 5% 13-gram overlap with eval tasks, MinHash near-duplicates removed |
| split | by source repository (`val_fraction` = `<x>`) |
| counts | from `training/data/out/stats.json`: runs, passed, kept, dropped by reason, train, val, DPO pairs |
| cost | teacher GPU hours × rate |
| format | JSONL, one conversation per line: `{"messages": [...], "tools": [...], "task_id": "..."}` (OpenAI chat format) |
| location | `<HF Hub private repo and revision>` |

## Known biases

- Mutation bugs are single-line and local; they teach "find the changed line",
  not design work.
- Rejection sampling keeps what the teacher already solves: the data is easier
  than the tasks it was drawn from.
