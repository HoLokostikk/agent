"""Command-line entry point: REPL, headless mode, and helper commands."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer
from dotenv import load_dotenv
from openai.types.chat import ChatCompletionMessageParam
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from agent.core import Agent, AgentEvents, Budget, RunResult, ToolCallRecord, load_prompt
from agent.core.builder import AgentSpec, build_agent
from agent.core.session import SessionStore
from agent.llm import (
    CallMetrics,
    ConfigError,
    LLMClient,
    LLMError,
    ModelConfig,
    ToolCall,
    get_model,
)
from agent.llm.config import load_models
from agent.permissions import Mode, PolicyError, load_rules
from agent.rag.search import Searcher
from agent.sandbox import DockerSandbox, Sandbox, SandboxError, build_image, image_exists
from agent.telemetry import DEFAULT_TRACE_FILE, Exporter, configure
from agent.tools import Tool, default_tools

DEFAULT_CONFIG = Path("configs/models.yaml")
DEFAULT_RULES = Path("configs/permissions.yaml")
DEFAULT_PROMPT = Path("configs/prompts/system.v1.md")

EXIT_OK = 0
EXIT_NOT_COMPLETED = 1
EXIT_USAGE = 2
EXIT_SANDBOX = 3
EXIT_MODEL_UNAVAILABLE = 4

app = typer.Typer(add_completion=False)
out = Console(highlight=False, soft_wrap=True)
err = Console(stderr=True, highlight=False)

ModelOpt = Annotated[str, typer.Option("--model", "-m", help="Model alias from the registry.")]
ConfigOpt = Annotated[Path, typer.Option("--config", help="Model registry file.")]


class Output(StrEnum):
    TEXT = "text"
    JSON = "json"


class SandboxKind(StrEnum):
    DOCKER = "docker"
    NONE = "none"  # bash disabled; file tools still work


# --- factories (replaced in tests) -------------------------------------------


def make_client(config: ModelConfig) -> LLMClient:
    return LLMClient(config)


@contextlib.asynccontextmanager
async def make_sandbox(root: Path, kind: SandboxKind) -> AsyncIterator[Sandbox | None]:
    if kind is SandboxKind.NONE:
        yield None
        return
    sandbox = DockerSandbox(root)
    if not await image_exists(sandbox.image):
        err.print(f"[dim]building sandbox image {sandbox.image} (first run only)...[/dim]")
        await build_image(sandbox.image)
    async with sandbox:
        yield sandbox


# --- formatting ----------------------------------------------------------------


def format_metrics(m: CallMetrics) -> str:
    parts: list[str] = []
    if m.usage is not None:
        tokens = f"{m.usage.input_tokens} in"
        if m.usage.cached_input_tokens:
            tokens += f" ({m.usage.cached_input_tokens} cached)"
        tokens += f" / {m.usage.output_tokens} out"
        if m.usage.reasoning_tokens:
            tokens += f" ({m.usage.reasoning_tokens} reasoning)"
        parts.append(tokens)
    else:
        parts.append("tokens unknown")
    parts.append(f"${m.cost_usd:.6f}" if m.cost_usd is not None else "$?")
    parts.append(f"latency {m.latency_s:.2f}s")
    parts.append(f"TTFT {m.ttft_s:.2f}s" if m.ttft_s is not None else "TTFT -")
    if m.output_tokens_per_s is not None:
        parts.append(f"{m.output_tokens_per_s:.1f} tok/s")
    if m.attempts > 1:
        parts.append(f"{m.attempts} attempts")
    parts.append(f"finish {m.finish_reason}")
    return " · ".join(parts)


def format_run(result: RunResult) -> str:
    s = result.stats
    cost = f"${s.cost_usd:.6f}" if s.cost_known else "$?"
    parts = [
        f"{result.stop_reason}",
        f"{s.turns} turns",
        f"{s.tool_calls} tool calls",
        f"{s.input_tokens} in / {s.output_tokens} out",
        cost,
        f"{s.wall_s:.1f}s",
    ]
    if s.cache_hit_rate is not None:
        parts.append(f"cache {s.cache_hit_rate:.0%}")
    if s.compactions:
        parts.append(f"{s.compactions} compactions")
    if s.invalid_tool_calls:
        parts.append(f"{s.invalid_tool_calls} invalid")
    if s.denied_tool_calls:
        parts.append(f"{s.denied_tool_calls} denied")
    return " · ".join(parts)


def _short_args(arguments: str, limit: int = 120) -> str:
    try:
        data = json.loads(arguments or "{}")
        text = (
            ", ".join(f"{k}={v!r}" for k, v in data.items())
            if isinstance(data, dict)
            else arguments
        )
    except json.JSONDecodeError:
        text = arguments
    return text if len(text) <= limit else text[: limit - 3] + "..."


class ConsoleEvents(AgentEvents):
    """Streams the conversation to the terminal."""

    def __init__(self, console: Console, show_reasoning: bool = False) -> None:
        self.console = console
        self.show_reasoning = show_reasoning
        self._mid_line = False

    def on_delta(self, kind: str, text: str) -> None:
        if kind == "content":
            self.console.print(text, end="", markup=False)
            self._mid_line = not text.endswith("\n")
        elif self.show_reasoning:
            self.console.print(text, end="", style="dim italic", markup=False)
            self._mid_line = not text.endswith("\n")

    def _newline(self) -> None:
        if self._mid_line:
            self.console.print()
            self._mid_line = False

    def on_tool_start(self, call: ToolCall) -> None:
        self._newline()
        self.console.print(
            f"[cyan]▸ {call.name}[/cyan] [dim]{_short_args(call.arguments)}[/dim]",
            markup=True,
            highlight=False,
        )

    def on_tool_end(self, record: ToolCallRecord) -> None:
        first = record.output.strip().splitlines()[0] if record.output.strip() else ""
        mark = "[green]✓[/green]" if record.ok else "[red]✗[/red]"
        self.console.print(f"  {mark} [dim]{_short(first)}[/dim]")


def _short(text: str, limit: int = 160) -> str:
    text = text.replace("[", "\\[")
    return text if len(text) <= limit else text[: limit - 3] + "..."


class ConsoleApprover:
    """Asks the human before writes and commands; 'a' trusts a tool for the session."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self.always: set[str] = set()

    async def __call__(self, tool: Tool, args: Any, preview: str | None) -> bool:
        if tool.name in self.always:
            return True
        if preview:
            lexer = "diff" if preview.startswith(("---", "(")) else "bash"
            self.console.print(Syntax(preview, lexer, theme="ansi_dark", word_wrap=True))
        answer = await asyncio.to_thread(
            input, f"Allow {tool.name}? [y]es / [n]o / [a]lways for {tool.name}: "
        )
        choice = answer.strip().lower()
        if choice in {"a", "always"}:
            self.always.add(tool.name)
            return True
        return choice in {"y", "yes"}


# --- agent session -------------------------------------------------------------


def _result_json(
    result: RunResult, agent: Agent, model: ModelConfig, prompt_version: str, mode: Mode, root: Path
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stop_reason": result.stop_reason.value,
        "completed": result.completed,
        "final_answer": result.final_answer,
        "error": result.error,
        "model": {"alias": model.alias, "model": model.model, "revision": model.revision},
        "prompt_version": prompt_version,
        "mode": mode.value,
        "workspace": str(root),
        "trace_id": result.trace_id,
        "stats": result.stats.as_dict(),
        "turns": [asdict(t) for t in result.turns],
        "messages": agent.messages,
    }


async def _agent_session(
    *,
    model: ModelConfig,
    root: Path,
    mode: Mode,
    sandbox_kind: SandboxKind,
    budget: Budget,
    prompt_path: Path,
    rules_path: Path,
    task: str | None,
    output: Output,
    show_reasoning: bool,
    compact_model: ModelConfig | None = None,
    compaction: bool = True,
    resume: str | None = None,
    sessions: SessionStore | None = None,
    rag: bool = False,
    embed_config: ModelConfig | None = None,
) -> int:
    prompt_version, system_prompt = load_prompt(prompt_path)
    searcher = await _build_searcher(root, embed_config) if rag else None
    spec = AgentSpec(
        base_prompt=system_prompt,
        prompt_version=prompt_version,
        rules=load_rules(rules_path),
        mode=mode,
        budget=budget,
        compaction=compaction,
        searcher=searcher,
    )
    client = make_client(model)
    compact_client = make_client(compact_model) if compact_model is not None else None
    store = sessions or SessionStore()
    interactive = task is None
    events: AgentEvents | None
    if output is Output.JSON:
        events = None
    else:
        events = ConsoleEvents(out if interactive else err, show_reasoning)
        if not interactive:
            # Headless text mode: tool activity on stderr, answer on stdout.
            events = _HeadlessTextEvents(err, show_reasoning)
    approver = ConsoleApprover(out) if interactive and mode is Mode.ASK else None
    try:
        async with make_sandbox(root, sandbox_kind) as sandbox:
            agent = build_agent(
                client,
                root,
                sandbox,
                spec,
                compact_model=compact_client,
                approver=approver,
                events=events,
                trace_attributes={"agent.session": "repl" if interactive else "headless"},
            )
            session_id = store.new_id()
            if resume is not None:
                meta, history = store.load(resume)
                session_id = meta["id"]
                # Keep the current system prompt; replay the rest of the conversation.
                agent.messages = [agent.messages[0], *history[1:]]
                err.print(f"[dim]resumed session {session_id} ({len(history) - 1} messages)[/dim]")
            meta_out = {
                "created": session_id[:15],
                "workspace": str(root),
                "model": model.alias,
                "prompt_version": prompt_version,
            }

            def save() -> None:
                store.save(session_id, agent.messages, meta_out)

            if task is not None:
                result = await agent.run(task)
                save()
                if output is Output.JSON:
                    payload = _result_json(result, agent, model, prompt_version, mode, root)
                    out.print_json(json.dumps(payload, default=str))
                else:
                    if result.final_answer:
                        out.print(result.final_answer, markup=False)
                    err.print(f"[dim]{format_run(result)}[/dim]")
                    if result.trace_id:
                        err.print(f"[dim]trace {result.trace_id}[/dim]")
                    if result.error:
                        err.print(f"[red]{result.error}[/red]")
                err.print(f"[dim]session {session_id} (agent --resume {session_id})[/dim]")
                return EXIT_OK if result.completed else EXIT_NOT_COMPLETED
            code = await _repl(agent, model, mode, sandbox is not None, save)
            err.print(f"[dim]session {session_id} (agent --resume {session_id})[/dim]")
            return code
    finally:
        await client.aclose()
        if compact_client is not None:
            await compact_client.aclose()


class _HeadlessTextEvents(ConsoleEvents):
    def on_delta(self, kind: str, text: str) -> None:
        if kind == "reasoning" and self.show_reasoning:
            super().on_delta(kind, text)


async def _repl(
    agent: Agent, model: ModelConfig, mode: Mode, has_sandbox: bool, save: Callable[[], None]
) -> int:
    sandbox_note = "docker sandbox" if has_sandbox else "no sandbox (bash disabled)"
    out.print(
        f"[bold]agent[/bold] · {model.alias} · mode {mode} · {sandbox_note} · /exit to quit",
    )
    total = 0.0
    while True:
        try:
            text = await asyncio.to_thread(input, "\n› ")
        except EOFError:
            break
        text = text.strip()
        if not text:
            continue
        if text in {"/exit", "/quit"}:
            break
        result = await agent.run(text)
        save()
        out.print()
        total += result.stats.cost_usd
        out.print(f"[dim]{format_run(result)}[/dim]")
        if result.trace_id:
            out.print(f"[dim]trace {result.trace_id}[/dim]")
        if result.error:
            out.print(f"[red]{result.error}[/red]")
    out.print(f"[dim]session cost ${total:.6f}[/dim]")
    return EXIT_OK


# --- commands ------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    prompt: Annotated[
        str | None, typer.Option("--prompt", "-p", help="Run one task headless and exit.")
    ] = None,
    output: Annotated[Output, typer.Option(help="Headless output format.")] = Output.TEXT,
    model: ModelOpt = "local-small",
    workspace: Annotated[Path, typer.Option(help="Repository to work in.")] = Path(),
    mode: Annotated[
        Mode | None, typer.Option(help="Permissions; default: ask in the REPL, auto headless.")
    ] = None,
    sandbox: Annotated[SandboxKind, typer.Option(help="Where bash runs.")] = SandboxKind.DOCKER,
    max_turns: Annotated[int, typer.Option(min=1)] = 30,
    max_cost: Annotated[float, typer.Option(min=0)] = 1.0,
    max_time: Annotated[float, typer.Option(min=1, help="Seconds.")] = 1800.0,
    config: ConfigOpt = DEFAULT_CONFIG,
    rules: Annotated[Path, typer.Option(help="Permission rules.")] = DEFAULT_RULES,
    system_prompt: Annotated[Path, typer.Option(help="System prompt file.")] = DEFAULT_PROMPT,
    show_reasoning: Annotated[bool, typer.Option(help="Print reasoning tokens.")] = False,
    compact_model: Annotated[
        str | None, typer.Option(help="Alias of a cheaper model for compaction and explorer.")
    ] = None,
    compaction: Annotated[bool, typer.Option(help="Summarise old turns near the limit.")] = True,
    resume: Annotated[
        str | None, typer.Option(help="Continue a saved session: its id, or 'last'.")
    ] = None,
    rag: Annotated[bool, typer.Option(help="Index the workspace and add search_code.")] = False,
    embed_model: Annotated[
        str | None, typer.Option(help="Embedding model alias for --rag (default: BM25 only).")
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Log every model call.")] = False,
    trace: Annotated[
        Exporter, typer.Option(envvar="AGENT_TRACE", help="Where spans go (otlp: Phoenix etc.).")
    ] = Exporter.FILE,
    trace_file: Annotated[
        Path, typer.Option(envvar="AGENT_TRACE_FILE", help="JSONL file for --trace file.")
    ] = DEFAULT_TRACE_FILE,
) -> None:
    """Terminal coding agent. No subcommand: interactive REPL; -p: one headless task."""
    provider = configure(trace, path=trace_file) if ctx.invoked_subcommand != "trace" else None
    if provider is not None:
        ctx.call_on_close(provider.shutdown)
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    if ctx.invoked_subcommand is not None:
        return
    root = workspace.resolve()
    if not root.is_dir():
        err.print(f"[red]workspace is not a directory:[/red] {root}")
        raise typer.Exit(EXIT_USAGE)
    effective_mode = mode or (Mode.AUTO if prompt is not None else Mode.ASK)
    try:
        model_config = get_model(config, model)
        compact_config = get_model(config, compact_model) if compact_model else None
        embed_config = get_model(config, embed_model) if embed_model else None
        code = asyncio.run(
            _agent_session(
                model=model_config,
                root=root,
                mode=effective_mode,
                sandbox_kind=sandbox,
                budget=Budget(max_turns=max_turns, max_cost_usd=max_cost, max_wall_s=max_time),
                prompt_path=system_prompt,
                rules_path=rules,
                task=prompt,
                output=output,
                show_reasoning=show_reasoning,
                compact_model=compact_config,
                compaction=compaction,
                resume=resume,
                rag=rag,
                embed_config=embed_config,
            )
        )
    except (ConfigError, PolicyError, FileNotFoundError) as exc:
        err.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(EXIT_USAGE) from None
    except SandboxError as exc:
        err.print(f"[red]sandbox error:[/red] {exc}")
        raise typer.Exit(EXIT_SANDBOX) from None
    raise typer.Exit(code)


async def _chat(
    config: ModelConfig, prompt: str | None, system: str | None, show_reasoning: bool
) -> int:
    client = make_client(config)
    history: list[ChatCompletionMessageParam] = []
    if system:
        history.append({"role": "system", "content": system})
    total_cost = 0.0

    def on_delta(kind: str, text: str) -> None:
        if kind == "content":
            out.print(text, end="", markup=False)
        elif show_reasoning:
            out.print(text, end="", style="dim", markup=False)

    try:
        while True:
            if prompt is not None:
                user_text = prompt
            else:
                try:
                    user_text = await asyncio.to_thread(input, "› ")
                except EOFError:
                    break
                if user_text.strip() in {"/exit", "/quit"}:
                    break
                if not user_text.strip():
                    continue
            history.append({"role": "user", "content": user_text})
            try:
                completion = await client.complete(history, on_delta=on_delta)
            except LLMError as exc:
                err.print(f"[red]model call failed:[/red] {exc}")
                return 1
            out.print()
            history.append(completion.to_message())
            total_cost += completion.metrics.cost_usd or 0.0
            err.print(f"[dim]{format_metrics(completion.metrics)}[/dim]")
            if prompt is not None:
                break
    finally:
        await client.aclose()
    if prompt is None:
        err.print(f"[dim]session cost ${total_cost:.6f}[/dim]")
    return 0


@app.command()
def chat(
    prompt: Annotated[str | None, typer.Argument(help="One-shot prompt; omit for a REPL.")] = None,
    model: ModelOpt = "local-small",
    config: ConfigOpt = DEFAULT_CONFIG,
    system: Annotated[str | None, typer.Option(help="System prompt.")] = None,
    show_reasoning: Annotated[bool, typer.Option(help="Print reasoning tokens.")] = False,
) -> None:
    """Plain chat with a model (no tools); prints tokens, cost, latency and TTFT per call."""
    try:
        model_config = get_model(config, model)
        code = asyncio.run(_chat(model_config, prompt, system, show_reasoning))
    except ConfigError as exc:
        err.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(EXIT_USAGE) from None
    raise typer.Exit(code)


@app.command()
def models(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """List model aliases in the registry."""
    try:
        registry = load_models(config)
    except ConfigError as exc:
        err.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(EXIT_USAGE) from None
    table = Table("alias", "model", "revision", "context", "$/1M in", "$/1M out")
    for alias, m in sorted(registry.items()):
        revision = m.revision if len(m.revision) <= 24 else m.revision[:21] + "..."
        table.add_row(
            alias,
            m.model,
            revision,
            str(m.context_window),
            f"{m.pricing.input_per_mtok:g}",
            f"{m.pricing.output_per_mtok:g}",
        )
    out.print(table)


@app.command("build-sandbox")
def build_sandbox() -> None:
    """Build the Docker image the bash tool runs in."""
    try:
        asyncio.run(build_image())
    except SandboxError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_SANDBOX) from None
    out.print("sandbox image built")


@app.command("sessions")
def sessions_cmd(limit: int = 20) -> None:
    """Saved sessions, newest first (resume with agent --resume <id>)."""
    table = Table("id", "model", "messages", "workspace")
    for info in SessionStore().list(limit):
        table.add_row(info.id, info.model, str(info.messages), info.workspace)
    out.print(table)


# --- retrieval -------------------------------------------------------------------


async def _build_searcher(root: Path, embed_config: ModelConfig | None) -> Searcher:
    from agent.rag.embed import OpenAIEmbedder
    from agent.rag.index import CodeIndex
    from agent.rag.search import Searcher

    embedder = OpenAIEmbedder(embed_config) if embed_config else None
    index = CodeIndex(root)
    try:
        stats = await index.update(embedder)
        chunks, vectors = index.vectors()
    finally:
        index.close()
    err.print(
        f"[dim]index: {stats.chunks_total} chunks, {stats.files_changed} files updated, "
        f"{stats.files_deleted} removed ({'hybrid' if vectors is not None else 'bm25 only'})[/dim]"
    )
    return Searcher(chunks, vectors, embedder)


@app.command("index")
def index_cmd(
    workspace: Annotated[Path, typer.Option()] = Path(),
    embed_model: Annotated[str | None, typer.Option()] = None,
    config: ConfigOpt = DEFAULT_CONFIG,
    query: Annotated[str | None, typer.Option(help="Run one search after indexing.")] = None,
) -> None:
    """Build or refresh the code index of a workspace (incremental)."""
    embed_config = get_model(config, embed_model) if embed_model else None

    async def go() -> None:
        searcher = await _build_searcher(workspace.resolve(), embed_config)
        if query:
            for hit in await searcher.search(query, 5):
                out.print(hit.chunk.header(), markup=False)

    asyncio.run(go())


# --- traces ----------------------------------------------------------------------

trace_app = typer.Typer(no_args_is_help=True, help="Inspect recorded traces.")
app.add_typer(trace_app, name="trace")
TraceFileOpt = Annotated[Path, typer.Option("--file", envvar="AGENT_TRACE_FILE")]


@trace_app.command("list")
def trace_list(file: TraceFileOpt = DEFAULT_TRACE_FILE, limit: int = 20) -> None:
    """Most recent agent runs with their trace ids."""
    import datetime as dt

    from agent.telemetry.view import runs

    table = Table("trace", "started", "stop reason", "turns", "tools", "cost", "duration")
    for span in runs(file, limit):
        a = span.attributes
        started = dt.datetime.fromtimestamp(span.start_ns / 1e9).strftime("%m-%d %H:%M")
        cost = a.get("agent.cost_usd")
        table.add_row(
            span.trace_id,
            started,
            str(a.get("agent.stop_reason", "?")),
            str(a.get("agent.turns", "?")),
            str(a.get("agent.tool_calls", "?")),
            "?" if cost is None else f"${float(cost):.4f}",
            f"{span.duration_s:.1f}s",
        )
    out.print(table)


@trace_app.command("show")
def trace_show(trace_id: str, file: TraceFileOpt = DEFAULT_TRACE_FILE) -> None:
    """One trace as a tree: run → turns → model calls and tool calls."""
    from agent.telemetry.view import render_tree

    out.print(render_tree(file, trace_id), markup=False)


@trace_app.command("stats")
def trace_stats(file: TraceFileOpt = DEFAULT_TRACE_FILE) -> None:
    """p50/p95 latency and TTFT per model, cost per run, tool failure rates."""
    from agent.telemetry.view import stats

    out.print_json(json.dumps(stats(file)))


# --- evals -----------------------------------------------------------------------

eval_app = typer.Typer(no_args_is_help=True, help="Run evals and compare results.")
app.add_typer(eval_app, name="eval")

DEFAULT_TASKS = Path("evals/tasks")
DEFAULT_CASES = Path("evals/tool_calls/cases.yaml")
DEFAULT_RESULTS = Path("evals/results")


async def _preflight(client: LLMClient) -> None:
    """One tiny call before a long eval, so a dead endpoint fails loudly instead
    of producing a results file full of zeros."""
    try:
        await client.complete([{"role": "user", "content": "Reply with: ok"}], max_tokens=2)
    except LLMError as exc:
        await client.aclose()
        err.print(f"[red]model endpoint is not usable, eval not started:[/red] {exc}")
        raise typer.Exit(EXIT_MODEL_UNAVAILABLE) from None


def _write_results(
    payload: dict[str, Any], out_dir: Path, kind: str, alias: str, version: str
) -> Path:
    from agent.evals.report import render
    from agent.evals.runner import results_path

    out_dir.mkdir(parents=True, exist_ok=True)
    path = results_path(out_dir, kind, alias, version)
    path.write_text(json.dumps(payload, indent=1, default=str) + "\n", encoding="utf-8")
    path.with_suffix(".md").write_text(render(path), encoding="utf-8")
    return path


@eval_app.command("tasks")
def eval_tasks(
    model: ModelOpt = "local-small",
    k: Annotated[int, typer.Option("-k", min=1, help="Runs per task.")] = 3,
    concurrency: Annotated[int, typer.Option(min=1)] = 1,
    tag: Annotated[list[str] | None, typer.Option(help="Only tasks with this tag.")] = None,
    task: Annotated[list[str] | None, typer.Option(help="Only this task id.")] = None,
    tasks_dir: Annotated[Path, typer.Option()] = DEFAULT_TASKS,
    out_dir: Annotated[Path, typer.Option("--out")] = DEFAULT_RESULTS,
    max_cost: Annotated[float, typer.Option(min=0, help="Per run, USD.")] = 1.0,
    max_time: Annotated[float, typer.Option(min=1, help="Per run, seconds.")] = 900.0,
    config: ConfigOpt = DEFAULT_CONFIG,
    rules: Annotated[Path, typer.Option()] = DEFAULT_RULES,
    system_prompt: Annotated[Path, typer.Option()] = DEFAULT_PROMPT,
) -> None:
    """Run the agent on golden tasks, k times each, and write a results file."""
    from agent.evals.runner import SuiteConfig, results_payload, run_suite, summarize
    from agent.evals.tasks import TaskError, load_tasks

    try:
        model_config = get_model(config, model)
        tasks = load_tasks(tasks_dir, tags=tag, ids=task)
        version, prompt_text = load_prompt(system_prompt)
        policy_rules = load_rules(rules)
    except (ConfigError, TaskError, PolicyError, FileNotFoundError) as exc:
        err.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(EXIT_USAGE) from None
    if not tasks:
        err.print("[red]no tasks selected[/red]")
        raise typer.Exit(EXIT_USAGE)

    from agent.evals.runner import git_revision

    git = git_revision(Path.cwd())

    async def go() -> Path:
        client = make_client(model_config)
        await _preflight(client)
        if not await image_exists():
            err.print("[dim]building sandbox image...[/dim]")
            await build_image()
        try:
            cfg = SuiteConfig(
                model=client,
                rules=policy_rules,
                system_prompt=prompt_text,
                prompt_version=version,
                k=k,
                concurrency=concurrency,
                budget=Budget(max_cost_usd=max_cost, max_wall_s=max_time),
            )
            records = await run_suite(
                tasks,
                cfg,
                progress=lambda r: err.print(
                    f"[dim]{r.task_id} #{r.attempt}: "
                    f"{'[green]pass[/green]' if r.passed else '[red]fail[/red]'} "
                    f"({r.stop_reason}, {r.stats.get('turns', 0)} turns, {r.wall_s:.0f}s)"
                    f"{' [yellow]tests edited[/yellow]' if r.tampered else ''}"
                    f"{' [red]' + r.error + '[/red]' if r.error else ''}[/dim]"
                ),
            )
            summary = summarize(records, k)
            payload = results_payload(
                "tasks",
                client,
                version,
                summary,
                [asdict(r) for r in records],
                {"tasks_dir": str(tasks_dir), "budget": asdict(cfg.budget)},
                git=git,
            )
            return _write_results(payload, out_dir, "tasks", model_config.alias, version)
        finally:
            await client.aclose()

    path = asyncio.run(go())
    out.print(path.with_suffix(".md").read_text(encoding="utf-8"), markup=False)
    err.print(f"[dim]results: {path}[/dim]")


@eval_app.command("toolcalls")
def eval_toolcalls(
    model: ModelOpt = "local-small",
    k: Annotated[int, typer.Option("-k", min=1, help="Runs per case.")] = 1,
    concurrency: Annotated[int, typer.Option(min=1)] = 1,
    cases: Annotated[Path, typer.Option()] = DEFAULT_CASES,
    out_dir: Annotated[Path, typer.Option("--out")] = DEFAULT_RESULTS,
    config: ConfigOpt = DEFAULT_CONFIG,
    system_prompt: Annotated[Path, typer.Option()] = DEFAULT_PROMPT,
) -> None:
    """Single-step tool-call eval: does the model pick the right tool and arguments?"""
    from agent.evals.runner import results_payload
    from agent.evals.toolcalls import CaseError, load_cases, run_cases, summarize_cases

    try:
        model_config = get_model(config, model)
        case_list = load_cases(cases)
        version, prompt_text = load_prompt(system_prompt)
    except (ConfigError, CaseError, FileNotFoundError) as exc:
        err.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(EXIT_USAGE) from None

    from agent.evals.runner import git_revision

    git = git_revision(Path.cwd())

    async def go() -> Path:
        client = make_client(model_config)
        await _preflight(client)
        try:
            results = await run_cases(
                case_list,
                client,
                default_tools(),
                prompt_text,
                k=k,
                concurrency=concurrency,
                progress=lambda r: err.print(f"[dim]{r.case_id} #{r.attempt}: {r.outcome}[/dim]"),
            )
            payload = results_payload(
                "toolcalls",
                client,
                version,
                summarize_cases(results),
                [asdict(r) for r in results],
                {"cases_file": str(cases)},
                git=git,
            )
            return _write_results(payload, out_dir, "toolcalls", model_config.alias, version)
        finally:
            await client.aclose()

    path = asyncio.run(go())
    out.print(path.with_suffix(".md").read_text(encoding="utf-8"), markup=False)
    err.print(f"[dim]results: {path}[/dim]")


@eval_app.command("retrieval")
def eval_retrieval(
    repo: Annotated[Path, typer.Option(help="A git repository to build queries from.")],
    embed_model: Annotated[str | None, typer.Option()] = None,
    max_commits: Annotated[int, typer.Option(min=1)] = 300,
    rerank: Annotated[bool, typer.Option(help="Cross-encoder rerank (needs torch).")] = False,
    out_dir: Annotated[Path, typer.Option("--out")] = DEFAULT_RESULTS,
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Retrieval ablation on git history: BM25 / vector / hybrid / hybrid+rerank."""
    import datetime as dt

    from agent.evals.retrieval import cases_from_git, evaluate, render_retrieval
    from agent.evals.runner import git_revision
    from agent.rag.embed import OpenAIEmbedder
    from agent.rag.index import CodeIndex
    from agent.rag.search import CrossEncoderReranker, Searcher

    embed_config = get_model(config, embed_model) if embed_model else None
    root = repo.resolve()
    cases = cases_from_git(root, max_commits)
    if not cases:
        err.print("[red]no usable commits (need commits that change 1-5 .py files)[/red]")
        raise typer.Exit(EXIT_USAGE)
    git = git_revision(Path.cwd())

    async def go() -> dict[str, Any]:
        embedder = OpenAIEmbedder(embed_config) if embed_config else None
        index = CodeIndex(root, root.parent / f".{root.name}-eval-index.sqlite")
        try:
            stats = await index.update(embedder)
            chunks, vectors = index.vectors()
        finally:
            index.close()
        searcher = Searcher(chunks, vectors, embedder, CrossEncoderReranker() if rerank else None)
        result = await evaluate(searcher, cases)
        return {
            "schema_version": 1,
            "kind": "retrieval",
            "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "git": git,
            "repo": root.name,
            "repo_revision": git_revision(root),
            "embedder": embedder.name if embedder else None,
            "chunks": stats.chunks_total,
            "files": stats.files_seen,
            **result,
        }

    payload = asyncio.run(go())
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H%M")
    path = out_dir / f"{stamp}_retrieval_{root.name}.json"
    path.write_text(json.dumps(payload, indent=1, default=str) + "\n", encoding="utf-8")
    report = render_retrieval(payload, path.name)
    path.with_suffix(".md").write_text(report, encoding="utf-8")
    out.print(report, markup=False)


@eval_app.command("report")
def eval_report(results: Path) -> None:
    """Print the markdown report for a results file."""
    from agent.evals.report import render

    out.print(render(results), markup=False)


@eval_app.command("compare")
def eval_compare(
    baseline: Path,
    candidate: Path,
    max_drop: Annotated[float, typer.Option(min=0, max=1, help="Allowed pass@1 drop.")] = 0.10,
) -> None:
    """Regression gate: exit 1 if candidate pass@1 drops more than --max-drop."""
    from agent.evals.report import gate, load_results

    result = gate(load_results(baseline), load_results(candidate), max_drop=max_drop)
    out.print(result.message)
    raise typer.Exit(0 if result.passed else 1)


def main() -> None:
    load_dotenv()
    app()
