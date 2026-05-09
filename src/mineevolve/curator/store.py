"""External knowledge store K (paper Section 3.3).

JSON-backed: skills.json + remedies.json. Atomic-ish writes via a temp file.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Dict, Iterable, List

from ..inducer.schema import KnowledgeEntry


class KnowledgeStore:
    """Append-and-merge external knowledge bank."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.root = Path(path)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._skills: Dict[str, KnowledgeEntry] = {}
        self._remedies: Dict[str, KnowledgeEntry] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _skills_path(self) -> Path:
        return self.root / "skills.json"

    def _remedies_path(self) -> Path:
        return self.root / "remedies.json"

    def _load(self) -> None:
        for p, target in (
            (self._skills_path(), self._skills),
            (self._remedies_path(), self._remedies),
        ):
            if not p.exists():
                continue
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                continue
            if not isinstance(data, list):
                continue
            for item in data:
                if not isinstance(item, dict):
                    continue
                entry = KnowledgeEntry.from_dict(item)
                target[entry.knowledge_id] = entry

    def _atomic_write(self, path: Path, payload: List[dict]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def flush(self) -> None:
        with self._lock:
            self._atomic_write(
                self._skills_path(),
                [e.to_dict() for e in self._skills.values()],
            )
            self._atomic_write(
                self._remedies_path(),
                [e.to_dict() for e in self._remedies.values()],
            )

    # ------------------------------------------------------------------
    # Container
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._skills) + len(self._remedies)

    def all(self) -> List[KnowledgeEntry]:
        return list(self._skills.values()) + list(self._remedies.values())

    def skills(self) -> List[KnowledgeEntry]:
        return list(self._skills.values())

    def remedies(self) -> List[KnowledgeEntry]:
        return list(self._remedies.values())

    def get(self, knowledge_id: str) -> KnowledgeEntry | None:
        return self._skills.get(knowledge_id) or self._remedies.get(knowledge_id)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def insert(self, entry: KnowledgeEntry) -> None:
        with self._lock:
            (self._skills if entry.is_skill() else self._remedies)[entry.knowledge_id] = entry

    def remove(self, knowledge_id: str) -> None:
        with self._lock:
            self._skills.pop(knowledge_id, None)
            self._remedies.pop(knowledge_id, None)

    def update_usage(self, knowledge_ids: Iterable[str]) -> None:
        with self._lock:
            for kid in knowledge_ids:
                e = self._skills.get(kid) or self._remedies.get(kid)
                if e is not None:
                    e.usage_count += 1

    # ------------------------------------------------------------------
    # Stats (used by server status endpoint)
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        return {
            "n_skills": len(self._skills),
            "n_remedies": len(self._remedies),
            "n_total": len(self),
        }
