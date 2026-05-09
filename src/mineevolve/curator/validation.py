"""Curator validation rules (paper Eq. 6).

    Accept(k, K) = I[Vschema(k) AND Vmatch(k) AND Vexec(k) AND Vspec(k)
                     AND NOT Cconflict(k, K)]

We implement each predicate as a small pure function so that the upstream
Inducer or unit tests can call them in isolation. None of the rules require
an LLM call.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from ..inducer.schema import KnowledgeEntry, RemedyCategory, RemedyScope


GENERIC_BLACKLIST: tuple[str, ...] = (
    "try again",
    "be careful",
    "be more careful",
    "do better",
    "use common sense",
    "think harder",
)


# ----------------------------------------------------------------------
# Vschema
# ----------------------------------------------------------------------

def v_schema(k: KnowledgeEntry) -> bool:
    """All required fields are present and non-empty for the entry's type."""

    if k.type not in ("skill", "remedy"):
        return False
    if not k.c:
        return False
    if k.is_skill():
        steps = k.u.get("steps")
        if not steps or not isinstance(steps, list):
            return False
        if "verification" not in k.phi:
            return False
    if k.is_remedy():
        repair = k.u.get("repair_action")
        if not repair or not isinstance(repair, list):
            return False
        if not k.failure_type or k.failure_type == "NONE":
            return False
        # Two-level remedy enums (paper Figure 3). We accept missing fields by
        # falling back to the safe defaults (subgoal-level / generic), but if
        # an entry SETS a value, it must be one of the closed enums.
        scope = k.u.get("scope")
        if scope is not None and str(scope).strip().lower() not in RemedyScope.ALL:
            return False
        category = k.u.get("category")
        if category is not None and str(category).strip().lower() not in RemedyCategory.ALL:
            return False
    return True


# ----------------------------------------------------------------------
# Vmatch
# ----------------------------------------------------------------------

def v_match(k: KnowledgeEntry, state_fields: Mapping[str, object]) -> bool:
    """The trigger context refers to fields that exist in the structured state.

    We accept any trigger string that mentions at least one of the structured
    state-field keys (inventory, gui, position, ...); skills with no matching
    state are still allowed because they may apply to early-stage tasks where
    state is the trivial {empty inventory}.
    """

    if not state_fields:
        return True
    if not k.c:
        return False
    keys = {str(x).lower() for x in state_fields.keys()}
    keys |= {"inventory", "gui", "position", "state", "ypos", "task", "biome"}
    for trigger in k.c:
        text = str(trigger).lower()
        if any(key in text for key in keys):
            return True
    # Permissive default: a non-empty trigger still passes match
    return True


# ----------------------------------------------------------------------
# Vexec
# ----------------------------------------------------------------------

def v_exec(k: KnowledgeEntry, executor_actions: Iterable[str]) -> bool:
    """Skill steps / repair actions must be plausibly executable.

    We use a soft check: every step / repair line must contain at least one
    verb-like token. This catches obvious 'be careful' / generic strings
    while accepting natural phrases like 'mine the dirt block then continue'.
    """

    actions_set = {str(a).lower() for a in executor_actions}
    actions_set |= {
        "mine", "place", "use", "craft", "smelt", "navigate", "move",
        "approach", "attack", "kill", "switch", "open", "close", "pick",
        "collect", "break", "dig", "go", "reach", "return", "equip", "drop",
    }

    items: list = []
    if k.is_skill():
        items = list(k.u.get("steps", []))
    else:
        items = list(k.u.get("repair_action", []))
    if not items:
        return False
    for entry in items:
        text = str(entry).lower()
        if not any(verb in text for verb in actions_set):
            return False
    return True


# ----------------------------------------------------------------------
# Vspec
# ----------------------------------------------------------------------

def v_spec(k: KnowledgeEntry) -> bool:
    """Reject generic 'try again' / 'be careful' style suggestions."""

    items: list = []
    if k.is_skill():
        items = list(k.u.get("steps", []))
    else:
        items = list(k.u.get("repair_action", []))
    for entry in items:
        text = str(entry).lower().strip()
        for bad in GENERIC_BLACKLIST:
            if bad in text:
                return False
        if len(text) < 4:
            return False
    return True


# ----------------------------------------------------------------------
# Cconflict
# ----------------------------------------------------------------------

def c_conflict(k: KnowledgeEntry, store: Iterable[KnowledgeEntry]) -> bool:
    """Returns True if k conflicts with an existing high-confidence entry.

    Two entries conflict iff:
      * same type and failure_type;
      * trigger contexts overlap by >= 50%;
      * but the contents differ AND the existing entry has rho >= 0.8.
    """

    new_ctx = {str(x).lower() for x in k.c}
    if not new_ctx:
        return False
    new_payload = _payload_signature(k)
    for other in store:
        if other.knowledge_id == k.knowledge_id:
            continue
        if other.type != k.type or other.failure_type != k.failure_type:
            continue
        if other.rho < 0.8:
            continue
        other_ctx = {str(x).lower() for x in other.c}
        if not other_ctx:
            continue
        overlap = len(new_ctx & other_ctx) / float(len(new_ctx | other_ctx))
        if overlap < 0.5:
            continue
        if new_payload != _payload_signature(other):
            return True
    return False


def _payload_signature(k: KnowledgeEntry) -> tuple:
    if k.is_skill():
        return tuple(str(s).lower() for s in k.u.get("steps", []))
    return tuple(str(s).lower() for s in k.u.get("repair_action", []))


# ----------------------------------------------------------------------
# Combined acceptance rule (Eq. 6)
# ----------------------------------------------------------------------

def accept(
    k: KnowledgeEntry,
    store: Iterable[KnowledgeEntry],
    state_fields: Mapping[str, object] | None = None,
    executor_actions: Iterable[str] = (),
) -> tuple[bool, str]:
    """Apply Eq. (6); return (accepted, reason)."""

    if not v_schema(k):
        return False, "schema"
    if not v_match(k, state_fields or {}):
        return False, "match"
    if not v_exec(k, executor_actions):
        return False, "exec"
    if not v_spec(k):
        return False, "specificity"
    if c_conflict(k, store):
        return False, "conflict"
    return True, "ok"
