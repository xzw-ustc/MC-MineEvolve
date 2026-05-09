"""Curator retrieval (paper Eq. 7).

    K^R_i = argmax_{S subset K} sum_{k in S} r(c_i, k)
            s.t.  sum_{k in S} tokens(k) <= B

We implement the relevance score r(c_i, k) as a simple lexical-overlap
weighted by the entry's confidence and a small bonus for recently used
knowledge. A token estimator approximates ``tokens(k)`` so we can enforce
the prompt budget B without calling a real tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

from ..inducer.schema import KnowledgeEntry


# ----------------------------------------------------------------------
# Token estimation (cheap proxy for tokens(k))
# ----------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough count: words + 0.25 per char of any digit/punctuation rich part."""

    if not text:
        return 0
    return max(1, int(len(str(text).split()) + len(text) // 6))


def entry_tokens(k: KnowledgeEntry) -> int:
    parts: list[str] = []
    parts.extend(str(c) for c in k.c)
    if k.is_skill():
        parts.extend(str(s) for s in k.u.get("steps", []))
        parts.extend(str(s) for s in k.u.get("preconditions", []))
    else:
        parts.extend(str(s) for s in k.u.get("repair_action", []))
        parts.append(str(k.u.get("risk_pattern", "")))
    parts.extend(str(p) for p in k.phi.values())
    return sum(estimate_tokens(p) for p in parts) + 8  # 8 tokens overhead


# ----------------------------------------------------------------------
# Relevance score r(c_i, k)
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalContext:
    task_goal: str
    current_subgoal: str = ""
    inventory_keys: tuple[str, ...] = ()
    recent_failure_types: tuple[str, ...] = ()

    def to_token_set(self) -> set[str]:
        toks: set[str] = set()
        for source in (self.task_goal, self.current_subgoal, *self.inventory_keys, *self.recent_failure_types):
            if not source:
                continue
            for tok in str(source).lower().replace("/", " ").replace("_", " ").split():
                if len(tok) > 2:
                    toks.add(tok)
        return toks


def relevance_score(ctx: RetrievalContext, k: KnowledgeEntry) -> float:
    ctx_tokens = ctx.to_token_set()
    if not ctx_tokens:
        return 0.0

    entry_tokens_set: set[str] = set()
    for c in k.c:
        for tok in str(c).lower().split():
            if len(tok) > 2:
                entry_tokens_set.add(tok)
    if k.is_skill():
        for s in k.u.get("steps", []):
            for tok in str(s).lower().split():
                if len(tok) > 2:
                    entry_tokens_set.add(tok)
    else:
        for s in k.u.get("repair_action", []):
            for tok in str(s).lower().split():
                if len(tok) > 2:
                    entry_tokens_set.add(tok)

    if not entry_tokens_set:
        return 0.0

    overlap = len(ctx_tokens & entry_tokens_set) / float(len(ctx_tokens | entry_tokens_set))
    failure_bonus = 0.0
    if k.is_remedy() and k.failure_type in ctx.recent_failure_types:
        failure_bonus = 0.25
    confidence_term = 0.5 + 0.5 * float(k.rho)
    return float(overlap * confidence_term + failure_bonus)


# ----------------------------------------------------------------------
# Budget-constrained retrieval (Eq. 7)
# ----------------------------------------------------------------------

def retrieve(
    store: Iterable[KnowledgeEntry],
    ctx: RetrievalContext,
    budget_tokens: int,
    top_k: int = 16,
) -> List[KnowledgeEntry]:
    """Greedy budget-constrained selection by relevance score."""

    candidates = list(store)
    scored = [(relevance_score(ctx, k), entry_tokens(k), k) for k in candidates]
    scored = [s for s in scored if s[0] > 0]
    scored.sort(key=lambda t: (-t[0], t[1]))

    selected: List[KnowledgeEntry] = []
    used = 0
    for _, toks, k in scored[: max(top_k, 1) * 4]:
        if used + toks > budget_tokens:
            continue
        selected.append(k)
        used += toks
        if len(selected) >= top_k:
            break
    return selected


def split_skills_remedies(entries: Sequence[KnowledgeEntry]) -> tuple[List[KnowledgeEntry], List[KnowledgeEntry]]:
    skills = [k for k in entries if k.is_skill()]
    remedies = [k for k in entries if k.is_remedy()]
    return skills, remedies


def retrieval_context_from_state(
    task_goal: str,
    current_subgoal: str,
    inventory: Mapping[str, int] | None,
    recent_failure_types: Sequence[str] = (),
) -> RetrievalContext:
    inv_keys = tuple(str(k) for k in (inventory or {}).keys())
    return RetrievalContext(
        task_goal=str(task_goal),
        current_subgoal=str(current_subgoal),
        inventory_keys=inv_keys,
        recent_failure_types=tuple(str(f) for f in recent_failure_types),
    )
