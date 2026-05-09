"""Success-rate monitor (one counter per task)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SuccessMonitor:
    counts: Dict[str, int] = field(default_factory=dict)
    attempts: Dict[str, int] = field(default_factory=dict)

    def record(self, task: str, success: bool) -> None:
        self.attempts[task] = self.attempts.get(task, 0) + 1
        if success:
            self.counts[task] = self.counts.get(task, 0) + 1

    def rate(self, task: str) -> float:
        n = self.attempts.get(task, 0)
        if n <= 0:
            return 0.0
        return self.counts.get(task, 0) / float(n)

    def overall(self) -> float:
        total_attempts = sum(self.attempts.values())
        if total_attempts <= 0:
            return 0.0
        total_success = sum(self.counts.values())
        return float(total_success) / float(total_attempts)
