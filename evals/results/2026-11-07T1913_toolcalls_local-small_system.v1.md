# toolcalls eval · local-small · system.v1

- source: `2026-11-07T1913_toolcalls_local-small_system.v1.json`
- created: 2026-11-07T19:13:59+00:00 · git `d6669df-dirty`
- model: `qwen2.5-3b-instruct-8k` · revision `sha256:155251c4b20a9a96f844f6ed01e45338b06fe4ad5c456660b58eee40d7dd7cea`
- sampling: `{"temperature": 0.7, "top_p": 0.8, "top_k": 20}`

30 cases, 30 runs.

| metric | value |
|---|---|
| accuracy (right tool and arguments) | 66.7% [50.0%, 83.3%] |
| invalid JSON rate | 0.0% |
| no tool call | 6.7% |

Outcomes: correct 20, no_call 2, wrong_args 5, wrong_tool 3

| case | outcome | tool | arguments |
|---|---|---|---|
| find-entrypoint | wrong_tool | read_file | `{"path":"main.py"}` |
| glob-python-files | wrong_tool | list_dir | `{"path":"*/"}` |
| grep-function-definition | no_call | — | `` |
| list-root | wrong_args | list_dir | `{}` |
| read-readme | no_call | — | `` |
| run-linter | wrong_args | bash | `{"command":"ruff ."}` |
| run-python-version | wrong_tool | read_file | `{"path":"python_version.py"}` |
| run-single-test-file | wrong_args | bash | `{"command":"bash tests/test_api.py"}` |
| run-tests | wrong_args | bash | `{"command":"make test"}` |
| write-new-file | wrong_args | write_file | `{"content":"print('hello')","path":"hello.py"}` |
