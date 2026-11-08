# toolcalls eval · local-small · system.v1

- source: `2026-11-07T2217_toolcalls_local-small_system.v1.json`
- created: 2026-11-07T22:17:51+00:00 · git `72f69e9-dirty`
- model: `qwen2.5-3b-instruct-8k` · revision `sha256:155251c4b20a9a96f844f6ed01e45338b06fe4ad5c456660b58eee40d7dd7cea`
- sampling: `{"temperature": 0.7, "top_p": 0.8, "top_k": 20}`

30 cases, 30 runs.

| metric | value |
|---|---|
| accuracy (right tool and arguments) | 73.3% [56.7%, 90.0%] |
| invalid JSON rate | 0.0% |
| no tool call | 3.3% |

Outcomes: correct 22, no_call 1, wrong_args 6, wrong_tool 1

| case | outcome | tool | arguments |
|---|---|---|---|
| find-entrypoint | wrong_args | grep | `{"include":null,"path":null,"ignore_case":true,"pattern":"def main\\(.*\\)\n"}` |
| glob-python-files | wrong_tool | list_dir | `{"path":"*/"}` |
| glob-yaml | wrong_args | glob | `{"pattern":"**/*.yaml","path":""}` |
| grep-function-definition | no_call | — | `` |
| read-first-lines | wrong_args | read_file | `{"limit":20,"path":"main.py","offset":0}` |
| run-linter | wrong_args | bash | `{"command":"ruff ."}` |
| run-single-test-file | wrong_args | bash | `{"command":"bash tests/test_api.py"}` |
| run-tests | wrong_args | bash | `{"command":"make test"}` |
