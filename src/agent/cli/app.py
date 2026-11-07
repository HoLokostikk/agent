"""Command-line entry point."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from openai.types.chat import ChatCompletionMessageParam
from rich.console import Console
from rich.table import Table

from agent.llm import CallMetrics, ConfigError, LLMClient, LLMError, ModelConfig, get_model
from agent.llm.config import load_models

DEFAULT_CONFIG = Path("configs/models.yaml")

app = typer.Typer(no_args_is_help=True, add_completion=False)
out = Console(highlight=False, soft_wrap=True)
err = Console(stderr=True, highlight=False)

ModelOpt = Annotated[str, typer.Option("--model", "-m", help="Model alias from the registry.")]
ConfigOpt = Annotated[Path, typer.Option("--config", help="Model registry file.")]


def make_client(config: ModelConfig) -> LLMClient:
    """Factory, so tests can swap the transport."""
    return LLMClient(config)


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


@app.callback()
def main_callback(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Log every call.")] = False,
) -> None:
    """Terminal coding agent on an open-weight model."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # Per-request transport lines duplicate our own llm_call record.
    logging.getLogger("httpx2").setLevel(logging.WARNING)


@app.command()
def chat(
    prompt: Annotated[str | None, typer.Argument(help="One-shot prompt; omit for a REPL.")] = None,
    model: ModelOpt = "local-small",
    config: ConfigOpt = DEFAULT_CONFIG,
    system: Annotated[str | None, typer.Option(help="System prompt.")] = None,
    show_reasoning: Annotated[bool, typer.Option(help="Print reasoning tokens.")] = False,
) -> None:
    """Chat with a model and print tokens, cost, latency and TTFT per call."""
    try:
        model_config = get_model(config, model)
        code = asyncio.run(_chat(model_config, prompt, system, show_reasoning))
    except ConfigError as exc:
        err.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(2) from None
    raise typer.Exit(code)


@app.command()
def models(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """List model aliases in the registry."""
    try:
        registry = load_models(config)
    except ConfigError as exc:
        err.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(2) from None
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


def main() -> None:
    load_dotenv()
    app()
