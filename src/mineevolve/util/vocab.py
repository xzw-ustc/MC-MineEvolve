"""Shared closed-vocabulary tables used by every LLM prompt.

Centralising the enumerations here guarantees that:

  * the planner / inducer / adaptor prompts all show the LLM the SAME
    closed sets, so its outputs round-trip through Curator validation
    (Eq. 6) without surprises;
  * adding a new failure type or task kind only requires editing ONE
    file (this one), not three independently maintained docstrings.

Each helper returns a Markdown-friendly bullet block ready to embed in a
system prompt.
"""

from __future__ import annotations

from typing import Iterable

from ..monitor.failure import FAILURE_TYPES


# ----------------------------------------------------------------------
# Failure type vocabulary (paper Figure 3 + monitor.failure.FAILURE_TYPES)
# ----------------------------------------------------------------------

FAILURE_TYPE_DESCRIPTIONS: dict[str, str] = {
    "NONE": "the subgoal succeeded; no failure to report.",
    "NAV_STUCK": "agent oscillated near the same XYZ for many ticks with no inventory gain (Figure 3: Navigation Stuck).",
    "TARGET_UNREACHABLE": "agent moved a long distance but never gained the named target item (Figure 3: Target Unreachable).",
    "GUI_FAIL": "an inventory or crafting GUI was open but produced no inventory change (Figure 3: GUI Blockage).",
    "MISSING_ITEM": "the agent named a target item but never gained it; usually a missing tool, table, fuel, or recipe ingredient (Figure 3: Missing Tool).",
    "RECIPE_FAIL": "a crafting recipe was attempted but yielded no item; recipe is invalid or ingredients incorrect.",
    "SMELT_FAIL": "smelting was attempted but yielded no item; missing fuel or wrong source.",
    "DEADLOCK": "the same failure type repeats across multiple distinct subgoals (cross-subgoal stuck loop).",
    "MOB_KILLED": "agent died during execution.",
    "TIMEOUT": "subgoal exceeded its per-step budget without success.",
    "UNKNOWN": "no specific signal matched.",
}


# ----------------------------------------------------------------------
# Subgoal vocabularies (planner / adaptor outputs)
# ----------------------------------------------------------------------

TASK_KIND_DESCRIPTIONS: dict[str, str] = {
    "mine": "destroy a block or kill a mob to gain an item (use the STEVE-1 executor).",
    "craft": "produce an item via the crafting GUI or 2x2/3x3 grid.",
    "smelt": "produce an item via a furnace.",
    "use": "right-click an item or block (e.g. eat food, place water).",
    "combat": "engage hostile mobs.",
    "move": "navigate to a coordinate or named location without expecting any item gain.",
    "wait": "no-op for a fixed timeout (used to let a passive process finish).",
}

EXECUTOR_HINT_DESCRIPTIONS: dict[str, str] = {
    "stevei": "STEVE-1 text-conditioned policy; the only executor that can navigate, mine, attack, or place blocks.",
    "mc_craft": "the crafting helper; opens a crafting GUI and polls inventory for the target item.",
    "mc_smelt": "the smelting helper; opens a furnace and polls inventory for the smelted item.",
    "wait": "no executor is invoked; the wrapper just consumes ticks.",
}

CHECK_TYPE_DESCRIPTIONS: dict[str, str] = {
    "inv_ge": "{type: inv_ge, item: <id>, n: <int>} - inventory has >= n of <id>.",
    "inv_lt": "{type: inv_lt, item: <id>, n: <int>} - inventory has < n of <id>.",
    "ypos_le": "{type: ypos_le, n: <int>} - agent Y position <= n (used for descent).",
    "ypos_ge": "{type: ypos_ge, n: <int>} - agent Y position >= n (used for ascent).",
    "path_clear": "{type: path_clear} - no block obstructs the agent's facing direction.",
    "gui_closed": "{type: gui_closed} - no GUI is open at end of subgoal.",
}


# ----------------------------------------------------------------------
# Confidence calibration table
# ----------------------------------------------------------------------

CONFIDENCE_CALIBRATION: tuple[tuple[float, str], ...] = (
    (0.95, "directly verified by inventory delta or block change in the supporting feedback (no ambiguity)."),
    (0.80, "well-supported by the segment but minor variation between runs is plausible."),
    (0.65, "consistent with the segment but only one supporting case observed."),
    (0.50, "plausible heuristic with weak grounding; should be refined when more evidence arrives."),
    (0.35, "speculative; the LLM is filling a gap with prior knowledge, not direct observation."),
)


# ----------------------------------------------------------------------
# Markdown rendering helpers (for embedding inside system prompts)
# ----------------------------------------------------------------------

def render_bullet_table(items: Iterable[tuple[str, str]]) -> str:
    return "\n".join(f"  - `{name}`: {desc}" for name, desc in items)


def render_failure_types(only: Iterable[str] | None = None) -> str:
    keys = list(only) if only is not None else list(FAILURE_TYPES)
    return render_bullet_table(
        (k, FAILURE_TYPE_DESCRIPTIONS.get(k, "(no description)")) for k in keys
    )


def render_task_kinds() -> str:
    return render_bullet_table(TASK_KIND_DESCRIPTIONS.items())


def render_executor_hints() -> str:
    return render_bullet_table(EXECUTOR_HINT_DESCRIPTIONS.items())


def render_check_types() -> str:
    return render_bullet_table(CHECK_TYPE_DESCRIPTIONS.items())


def render_remedy_scopes() -> str:
    from ..inducer.schema import RemedyScope

    explanations = {
        RemedyScope.SUBGOAL_LOCAL: "modify only the immediate next subgoal (smallest blast radius).",
        RemedyScope.UNFINISHED_PLAN_SUFFIX: "may modify any subgoal AFTER the frozen prefix (default).",
        RemedyScope.TASK_GLOBAL: "may insert NEW subgoals at the head of the unfinished suffix (largest blast radius).",
    }
    return render_bullet_table((s, explanations[s]) for s in RemedyScope.ALL)


def render_remedy_categories() -> str:
    from ..inducer.schema import RemedyCategory

    explanations = {
        RemedyCategory.GENERIC: "subgoal-level fix expressed as trigger + risk + repair (default).",
        RemedyCategory.MISSING_PREREQUISITE: "task-level: insert obtain/place X before retrying the recipe.",
        RemedyCategory.DEADLOCK_PATTERN: "task-level: break a cross-subgoal stuck loop with a state-changing action.",
        RemedyCategory.RECOVERY_STRATEGY: "task-level: a multi-step recovery sequence (e.g. surface, restock, then descend).",
    }
    return render_bullet_table((c, explanations[c]) for c in RemedyCategory.ALL)


def render_confidence_calibration() -> str:
    return "\n".join(f"  - {value:.2f}: {desc}" for value, desc in CONFIDENCE_CALIBRATION)


__all__ = [
    "FAILURE_TYPE_DESCRIPTIONS",
    "TASK_KIND_DESCRIPTIONS",
    "EXECUTOR_HINT_DESCRIPTIONS",
    "CHECK_TYPE_DESCRIPTIONS",
    "CONFIDENCE_CALIBRATION",
    "render_bullet_table",
    "render_failure_types",
    "render_task_kinds",
    "render_executor_hints",
    "render_check_types",
    "render_remedy_scopes",
    "render_remedy_categories",
    "render_confidence_calibration",
]
