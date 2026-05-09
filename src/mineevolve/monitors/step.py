"""Per-task env-step counter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class StepMonitor:
    totals: Dict[str, int] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)

    def record(self, task: str, steps: int) -> None:
        self.totals[task] = self.totals.get(task, 0) + int(steps)
        self.counts[task] = self.counts.get(task, 0) + 1

    def average(self, task: str) -> float:
        n = self.counts.get(task, 0)
        if n <= 0:
            return 0.0
        return self.totals.get(task, 0) / float(n)
