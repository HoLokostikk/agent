# toolcalls eval · local-small · system.v1

- source: `2026-11-07T2219_toolcalls_local-small_system.v1.json`
- created: 2026-11-07T22:19:14+00:00 · git `72f69e9-dirty`
- model: `qwen2.5-3b-instruct-8k` · revision `sha256:155251c4b20a9a96f844f6ed01e45338b06fe4ad5c456660b58eee40d7dd7cea`
- sampling: `{"temperature": 0.7, "top_p": 0.8, "top_k": 20}`

30 cases, 30 runs.

| metric | value |
|---|---|
| accuracy (right tool and arguments) | 76.7% [60.0%, 90.0%] |
| invalid JSON rate | 0.0% |
| no tool call | 3.3% |

Outcomes: correct 23, no_call 1, wrong_args 5, wrong_tool 1

| case | outcome | tool | arguments |
|---|---|---|---|
| glob-python-files | wrong_tool | list_dir | `{"path":"/**"}` |
| glob-yaml | wrong_args | glob | `{"pattern":"**/*.yaml","path":""}` |
| grep-function-definition | no_call | — | `` |
| read-first-lines | wrong_args | read_file | `{"path":"main.py","offset":0,"limit":20}` |
| run-linter | wrong_args | bash | `{"command":"ruff ."}` |
| run-tests | wrong_args | bash | `{"command":"make test"}` |
| write-new-file | wrong_args | write_file | `{"content":"print('hello')","path":"hello.py"}` |
