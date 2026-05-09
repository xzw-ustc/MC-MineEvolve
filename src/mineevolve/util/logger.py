"""Result table + standard logging setup.

Uses ``rich.Table`` to print per-task and aggregate metrics. The format is
ours, written from scratch.
"""

from __future__ import annotations

import logging
from typing import Iterable, Mapping

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from ..monitors import StepMonitor, SuccessMonitor


_console = Console()


def setup_logging(level: int = logging.INFO) -> None:
    handler = RichHandler(console=_console, rich_tracebacks=True, show_path=False)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
    )


def print_results(
    title: str,
    tasks: Iterable[str],
    success: SuccessMonitor,
    step: StepMonitor,
) -> None:
    table = Table(title=title)
    table.add_column("Task", overflow="fold")
    table.add_column("Attempts", justify="right")
    table.add_column("Successes", justify="right")
    table.add_column("Success Rate", justify="right")
    table.add_column("Avg Steps", justify="right")

    tasks = list(tasks)
    for task in tasks:
        attempts = success.attempts.get(task, 0)
        successes = success.counts.get(task, 0)
        rate = success.rate(task)
        avg_steps = step.average(task)
        table.add_row(
            str(task),
            str(attempts),
            str(successes),
            f"{rate * 100:.2f}%",
            f"{avg_steps:.1f}" if avg_steps else "-",
        )

    if tasks:
        table.add_section()
        table.add_row(
            "[bold]Overall",
            str(sum(success.attempts.get(t, 0) for t in tasks)),
            str(sum(success.counts.get(t, 0) for t in tasks)),
            f"[bold]{success.overall() * 100:.2f}%",
            "",
        )
    _console.print(table)


def info_panel(message: str) -> None:
    _console.rule(message)
