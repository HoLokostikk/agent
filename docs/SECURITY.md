# Security model

## Assets

- The user's code and working tree.
- Secrets on the machine: `.env` files, API keys, SSH keys, cloud credentials.
- The host machine itself.
- The model provider account (spend).

## Who might attack

- A repository the agent is asked to work on (its files, comments, test output).
- An MCP server, either malicious or compromised after it was trusted.
- Any text the agent reads that someone else wrote.

The core problem is that a language model cannot reliably tell instructions
from data. Anything the agent reads may try to steer it. The design therefore
assumes the model *can* be steered, and limits what a steered model can do.

## Mitigations, by layer

| layer | mitigation | where |
|---|---|---|
| execution | every model-requested command runs in a Docker container: no network, all capabilities dropped, read-only root filesystem, host uid, memory/CPU/pid limits, the workspace as the only mount | `src/agent/sandbox/` |
| files | file tools resolve symlinks and refuse anything outside the workspace | `src/agent/tools/paths.py` |
| secrets | secret-looking files are never read or searched by tools and never indexed | `src/agent/tools/files.py`, `src/agent/rag/index.py` |
| permissions | read-only / ask / auto modes; deny rules in every mode; protected paths; allow-list defeated by shell metacharacters | `src/agent/permissions/`, `configs/permissions.yaml` |
| human in the loop | in ask mode writes and commands need approval, with a diff | `src/agent/cli/app.py` |
| MCP | namespaced tools, per-server permission kind and allowlist, description pinning (a changed description disables the tool), output wrapped as untrusted | `src/agent/mcp/client.py` |
| terminal | escape sequences and control characters stripped from tool output and model text | `src/agent/tools/sanitize.py` |
| telemetry | no message content in spans; attribute values redacted | `src/agent/telemetry/` |
| recovery | `/undo` restores files changed by the last request | `src/agent/tools/checkpoint.py` |

Breaking the "lethal trifecta" (private data + untrusted content + a way out):
the sandbox has no network, so even a fully steered agent has no exfiltration
channel through commands; secrets are kept out of what the tools read.

## Known gaps

- Changes made through `bash` inside the sandbox are not covered by `/undo`.
- The workspace mount is read-write: in auto mode a steered agent can damage
  the working tree (not the host). Use ask mode on repositories you care about,
  or run on a copy.
- Deny rules are best effort; the sandbox is the boundary, not the rules.
- Prompt-injection resistance and the false-positive rate of the command rules
  have not been measured yet (see `docs/status.md`).

## Reporting

Open a private security advisory on the GitHub repository.
