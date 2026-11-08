# Runbook

## Release a new adapter (shadow → canary → full)

1. Serve the candidate next to the current one:
   `LORA="ft-v1=... ft-v2=..." bash gpu/pod_serve.sh`.
2. **Shadow:** run `agent eval tasks -m <candidate alias> -k 3` and the tool-call
   eval on the same commit as the current baseline. Users never see it.
3. Gate: `agent eval compare <current> <candidate> --max-drop <from evals/baselines.yaml>`.
4. **Canary:** point a second alias (e.g. `agent-canary`) at the candidate in
   `serving/litellm/config.yaml`, route a share of sessions to it, watch
   `agent trace stats` (failure rates, latency) for a day.
5. **Full:** point `agent-main` at the candidate; reload the gateway.

## Roll back (target: under one minute)

1. Edit `serving/litellm/config.yaml`: point `agent-main` back to the previous model.
2. Reload: restart `litellm --config serving/litellm/config.yaml`.
3. Verify: `agent chat -m <alias using the gateway> "ok?"` answers; note the time.

No code is deployed; the alias is configuration. Rollback drill timing: **not
measured yet** (needs the GPU deployment).

## Incidents

| symptom | first check | action |
|---|---|---|
| every eval run is `model_error` | `curl $BASE_URL/models`; the eval preflight message | restart vLLM / tunnel; evals refuse to start against a dead endpoint |
| agent deleted or broke files | `/undo` in the REPL (edits via edit/write tools) | for bash-made changes: `git checkout` / restore from the copy; run in ask mode next time |
| a secret appeared in output | `agent trace show <id>`: spans carry no content | rotate the secret; add the file pattern to `SECRET_PATTERNS` in `src/agent/tools/files.py` |
| MCP tool disabled with "changed its description" | `agent mcp list` | read the new description; if acceptable, `agent mcp trust <server>` |
| answers look made up | trace: did it read files before answering? | add the case to the golden set (`docs/failure_taxonomy.md`) |

## Ownership

Single maintainer project: the author is on call for everything. Alerts are
not wired; `agent trace stats` is the dashboard.
