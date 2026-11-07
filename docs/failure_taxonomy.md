# Failure taxonomy

Classes of agent failure seen in real trajectories. Each class should end up
with golden-set cases that reproduce it. Counts and run ids live in
`EXPERIMENTS.md`; this file is the catalogue.

| # | class | first seen | layer | status |
|---|---|---|---|---|
| 1 | **Tool call dropped from history by the server template.** The stock Ollama qwen2.5 template renders an assistant turn as text *or* tool calls; a turn with both loses the call, the model imitates text-only turns and stops calling tools. | phase 2, run 1 | serving | fixed: `configs/ollama/` |
| 2 | **Tool call not parsed at all.** Moving `<tool_call>` inside `range .ToolCalls` in the template makes Ollama stop recognising tool calls; they arrive as plain text. | phase 2, template fix v1 | serving | fixed |
| 3 | **Empty reply.** Output tokens are generated but neither text nor a tool call reaches the client (a malformed call the server swallowed). The loop treated it as "done". | phase 2, series 1 (3/5 runs) | harness | fixed: nudge, `empty_response` stop |
| 4 | **Empty path argument.** `list_dir(path="")`, then the identical failing call again until loop detection fires. | phase 2, runs 5, 7 | tools (ACI) | fixed: `""` means the root |
| 5 | **Hallucinated file name on the first turn** (`test_code.py`, `solution.py`). | phase 2 | model / prompt | mitigated: prompt says start with `list_dir` |
| 6 | **Diagnosis without a fix.** The model states the bug correctly and then answers instead of editing. | phase 2, series 1 (2/5) | model | open |
| 7 | **Debugging by print.** Replaces the return value with a `print`, breaking the function. | phase 2, run 6 | model | open |
| 8 | **`edit_file` old_string mismatch, repeated.** Re-sends an `old_string` that is not in the file after being told so. | phase 2, series 2 | model / ACI | open: candidate fix is returning the closest matching lines |
| 9 | **Indentation broken by an edit** (tabs mixed into a 4-space file), turning a test failure into an import error. | phase 2, run 6 | model | open |
