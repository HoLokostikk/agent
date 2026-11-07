"""Command-line entry point: REPL, headless mode, and helper commands."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
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
from agent.permissions import Mode, PermissionPolicy, PolicyError, load_rules
from agent.sandbox import DockerSandbox, Sandbox, SandboxError, build_image, image_exists
from agent.tools import PathGuard, Tool, ToolContext, default_tools

DEFAULT_CONFIG = Path("configs/models.yaml")
DEFAULT_RULES = Path("configs/permissions.yaml")
DEFAULT_PROMPT = Path("configs/prompts/system.v1.md")

EXIT_OK = 0
EXIT_NOT_COMPLETED = 1
EXIT_USAGE = 2
EXIT_SANDBOX = 3

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
) -> int:
    prompt_version, system_prompt = load_prompt(prompt_path)
    policy = PermissionPolicy(mode, load_rules(rules_path))
    client = make_client(model)
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
            agent = Agent(
                client,
                default_tools(),
                ToolContext(PathGuard(root), sandbox),
                policy,
                system_prompt=system_prompt,
                budget=budget,
                approver=approver,
                events=events,
            )
            if task is not None:
                result = await agent.run(task)
                if output is Output.JSON:
                    payload = _result_json(result, agent, model, prompt_version, mode, root)
                    out.print_json(json.dumps(payload, default=str))
                else:
                    if result.final_answer:
                        out.print(result.final_answer, markup=False)
                    err.print(f"[dim]{format_run(result)}[/dim]")
                    if result.error:
                        err.print(f"[red]{result.error}[/red]")
                return EXIT_OK if result.completed else EXIT_NOT_COMPLETED
            return await _repl(agent, model, mode, sandbox is not None)
    finally:
        await client.aclose()


class _HeadlessTextEvents(ConsoleEvents):
    def on_delta(self, kind: str, text: str) -> None:
        if kind == "reasoning" and self.show_reasoning:
            super().on_delta(kind, text)


async def _repl(agent: Agent, model: ModelConfig, mode: Mode, has_sandbox: bool) -> int:
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
        out.print()
        total += result.stats.cost_usd
        out.print(f"[dim]{format_run(result)}[/dim]")
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
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Log every model call.")] = False,
) -> None:
    """Terminal coding agent. No subcommand: interactive REPL; -p: one headless task."""
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


def main() -> None:
    load_dotenv()
    app()
