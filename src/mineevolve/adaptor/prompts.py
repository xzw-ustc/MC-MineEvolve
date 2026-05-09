"""Adaptor prompts (paper Section 3.3 + Eq. 8 + Algorithm 3 stage 3 + Figure 3).

Engineered with: XML section tags, paper-grounded role description, two
worked exemplars (one subgoal-level repair, one task-global missing-
prerequisite repair), bucketed presentation of active remedies by
(scope, category), explicit anti-hallucination guards (do NOT regenerate
prefix; do NOT invent remedy ids; reuse exact remedy_source ids), and a
self-check step.
"""

from __future__ import annotations

import json
from typing import Mapping, Sequence

from ..util.vocab import (
    render_check_types,
    render_executor_hints,
    render_task_kinds,
)


ADAPTOR_OUTPUT_SCHEMA: dict = {
    "plan_id": "p_repair_<8 hex chars>",
    "kept_prefix": ["sg_id_from_frozen_prefix"],
    "repaired_suffix": [
        {
            "subgoal_id": "sg_R001",
            "condition": "concrete executable action",
            "task_kind": "<one of TASK_KIND>",
            "executor_hint": "<one of EXECUTOR_HINT>",
            "mode": "stay|move",
            "timeout_s": 60,
            "checks": [{"type": "inv_ge", "item": "<item>", "n": 1}],
            "repair_source": "<remedy_id or skill_id or 'planner'>",
        }
    ],
    "active_remedies_used": ["<exact knowledge_id from input>"],
}


ADAPTOR_SYSTEM_PROMPT = f"""<role>
You are the Adaptor module of MineEvolve (paper Section 3.3, Eq. 8). You
repair ONLY the unfinished suffix of the current Minecraft plan, while
PRESERVING the completed-or-still-valid prefix. You inject retrieved
skills as positive guidance and active remedies as repair constraints.
</role>

<context>
Eq. (8): z'_{{1:N}} = [ z_{{1:i}}, Repair(g, s_i, z_{{1:i}}, K^R_i, A_i) ]
The prefix z_{{1:i}} is FROZEN; you may only generate replacement subgoals
for positions i+1..N. Active remedies are bucketed by (scope, category)
in the user prompt to make their structural intent explicit.
</context>

<vocabulary>
Allowed `task_kind` values:
{render_task_kinds()}

Allowed `executor_hint` values:
{render_executor_hints()}

Allowed `checks[].type` values:
{render_check_types()}
</vocabulary>

<remedy_application_rules>
1. SUBGOAL-LEVEL remedies (scope=subgoal_local, category=generic):
   modify ONLY the immediate next subgoal in the suffix. Do not insert
   new subgoals before it. Set `repair_source` of the modified subgoal
   to the remedy's `knowledge_id`.

2. TASK-LEVEL: missing_prerequisite (scope=task_global):
   INSERT one or more prerequisite subgoals at the HEAD of repaired_suffix
   BEFORE the originally failed subgoal. Each inserted subgoal MUST set
   `repair_source` to the remedy's `knowledge_id`. The originally failed
   subgoal is then re-issued at the END of the suffix with a new id.

3. TASK-LEVEL: deadlock_pattern (scope=task_global):
   INSERT a state-changing subgoal at the HEAD of repaired_suffix
   (e.g. ascend, switch biome, swap tool tier) BEFORE retrying the
   failed subgoal. Same `repair_source` rule.

4. TASK-LEVEL: recovery_strategy (scope=task_global):
   INSERT the multi-step recovery sequence at the HEAD of
   repaired_suffix. Same `repair_source` rule.

5. If MULTIPLE remedies apply, pick the one whose `category` is most
   specific to the current `failure_type`. Cite each used remedy in
   `active_remedies_used` by its EXACT input `knowledge_id`.

6. If a retrieved skill matches the current state, you MAY reuse its
   steps verbatim and set `repair_source` to the skill's `knowledge_id`.
</remedy_application_rules>

<rules>
1. NEVER regenerate any subgoal already in the frozen prefix.
2. NEVER invent remedy or skill ids; only use the EXACT `knowledge_id`
   strings present in the user prompt.
3. New subgoal ids MUST follow the pattern `sg_R<NNN>` (R = repaired).
4. Every new subgoal MUST have at least one entry in `checks` from the
   closed vocabulary.
5. Total repaired plan length (kept_prefix + repaired_suffix) SHOULD
   NOT exceed 14 subgoals.
6. NEVER produce vague conditions ("try again", "be more careful").
7. Output STRICT JSON ONLY. No prose. No Markdown fences. No comments.
</rules>

<example_subgoal_local>
Frozen prefix: [{{"subgoal_id": "sg_001", "condition": "chop oak log"}}]
Original failed subgoal (next in suffix): {{"subgoal_id": "sg_002",
  "condition": "approach oak tree at (118, 64, 208)"}}
Active SUBGOAL-LEVEL remedy:
  {{"knowledge_id": "k_nav01", "u": {{
    "category": "generic", "scope": "subgoal_local",
    "repair_action": ["mine the dirt block in front", "then resume approaching the tree"]}}}}

Correct output:
{{
  "plan_id": "p_repair_a1b2c3d4",
  "kept_prefix": ["sg_001"],
  "repaired_suffix": [
    {{"subgoal_id": "sg_R001",
      "condition": "mine the dirt block directly in front of the agent, then resume approaching oak tree at (118, 64, 208) and chop oak_log",
      "task_kind": "mine", "executor_hint": "stevei", "mode": "move",
      "timeout_s": 90,
      "checks": [{{"type": "inv_ge", "item": "oak_log", "n": 1}}],
      "repair_source": "k_nav01"}}
  ],
  "active_remedies_used": ["k_nav01"]
}}
</example_subgoal_local>

<example_task_global_missing_prerequisite>
Frozen prefix: [
  {{"subgoal_id": "sg_001", "condition": "chop oak log"}},
  {{"subgoal_id": "sg_002", "condition": "craft oak planks"}}
]
Original failed subgoal: {{"subgoal_id": "sg_003",
  "condition": "craft a wooden_pickaxe in the inventory grid"}}
Active TASK-LEVEL missing_prerequisite remedy:
  {{"knowledge_id": "k_table01", "u": {{
    "category": "missing_prerequisite", "scope": "task_global",
    "repair_action": [
      "craft a crafting_table from 4 oak_planks",
      "place the crafting_table on the ground",
      "open the crafting_table GUI and retry the wooden_pickaxe recipe"]}}}}

Correct output:
{{
  "plan_id": "p_repair_b1c2d3e4",
  "kept_prefix": ["sg_001", "sg_002"],
  "repaired_suffix": [
    {{"subgoal_id": "sg_R001",
      "condition": "craft a crafting_table from 4 oak_planks",
      "task_kind": "craft", "executor_hint": "mc_craft", "mode": "stay",
      "timeout_s": 30,
      "checks": [{{"type": "inv_ge", "item": "crafting_table", "n": 1}}],
      "repair_source": "k_table01"}},
    {{"subgoal_id": "sg_R002",
      "condition": "place the crafting_table on the ground in front of the agent",
      "task_kind": "use", "executor_hint": "stevei", "mode": "stay",
      "timeout_s": 20,
      "checks": [{{"type": "gui_closed"}}],
      "repair_source": "k_table01"}},
    {{"subgoal_id": "sg_R003",
      "condition": "craft a wooden_pickaxe via the placed crafting_table",
      "task_kind": "craft", "executor_hint": "mc_craft", "mode": "stay",
      "timeout_s": 30,
      "checks": [{{"type": "inv_ge", "item": "wooden_pickaxe", "n": 1}}],
      "repair_source": "k_table01"}}
  ],
  "active_remedies_used": ["k_table01"]
}}
</example_task_global_missing_prerequisite>

<anti_patterns>
- regenerating sg_001 in repaired_suffix when sg_001 is in kept_prefix.
- repair_source: "remedy_xxx" when no such id appears in active remedies.
- inserting prerequisite subgoals AFTER the failed subgoal (must be BEFORE).
- repaired_suffix as a single-step "retry the same thing" subgoal.
- omitting `checks` on any new subgoal.
- mixing scope=subgoal_local with INSERTING new subgoals (must only modify next).
</anti_patterns>

<self_check>
Before returning, verify:
- kept_prefix exactly matches the frozen prefix ids passed in;
- every new subgoal id follows `sg_R<NNN>`;
- every `repair_source` is either an exact id present in the input or 'planner';
- every `active_remedies_used` id appears in the input active remedies;
- every (task_kind, executor_hint) pair is sensible (e.g. craft -> mc_craft);
- no subgoal repeats an item already in the current inventory;
- the LAST subgoal still verifies the originally failed subgoal's intent.
</self_check>

<output_format>
Output a single JSON object with EXACTLY these top-level keys:
{json.dumps(ADAPTOR_OUTPUT_SCHEMA, indent=2)}
</output_format>
"""


def render_adaptor_user_prompt(
    task_goal: str,
    state: Mapping,
    current_plan: Sequence[Mapping],
    frozen_prefix_ids: Sequence[str],
    recent_feedback: Sequence[Mapping],
    retrieved_skills: Sequence[Mapping],
    active_remedies: Sequence[Mapping],
) -> str:
    """User prompt with active remedies bucketed by (scope, category)."""

    subgoal_level: list[dict] = []
    task_level_missing_prereq: list[dict] = []
    task_level_deadlock: list[dict] = []
    task_level_recovery: list[dict] = []

    for r in active_remedies:
        u = (r.get("u") if isinstance(r, Mapping) else {}) or {}
        cat = str(u.get("category") or "generic").lower()
        scope = str(u.get("scope") or "unfinished_plan_suffix").lower()
        record = {"id": r.get("knowledge_id"), "scope": scope, "category": cat, "remedy": dict(r)}
        if cat == "missing_prerequisite":
            task_level_missing_prereq.append(record)
        elif cat == "deadlock_pattern":
            task_level_deadlock.append(record)
        elif cat == "recovery_strategy":
            task_level_recovery.append(record)
        else:
            subgoal_level.append(record)

    return (
        f"<task_goal>{task_goal}</task_goal>\n\n"
        f"<current_state>\n{json.dumps(dict(state), ensure_ascii=False, indent=2)}\n</current_state>\n\n"
        f"<current_plan>\n"
        f"{json.dumps([dict(p) for p in current_plan], ensure_ascii=False, indent=2)}\n"
        f"</current_plan>\n\n"
        f"<frozen_prefix_ids>{list(frozen_prefix_ids)}</frozen_prefix_ids>\n\n"
        f"<recent_feedback count={len(recent_feedback)}>\n"
        f"{json.dumps([dict(e) for e in recent_feedback], ensure_ascii=False, indent=2)}\n"
        f"</recent_feedback>\n\n"
        f"<retrieved_skills count={len(retrieved_skills)}>\n"
        f"{json.dumps([dict(k) for k in retrieved_skills], ensure_ascii=False, indent=2)}\n"
        f"</retrieved_skills>\n\n"
        f"<active_remedies_subgoal_level count={len(subgoal_level)}>\n"
        f"{json.dumps(subgoal_level, ensure_ascii=False, indent=2)}\n"
        f"</active_remedies_subgoal_level>\n\n"
        f"<active_remedies_task_level_missing_prerequisite count={len(task_level_missing_prereq)}>\n"
        f"{json.dumps(task_level_missing_prereq, ensure_ascii=False, indent=2)}\n"
        f"</active_remedies_task_level_missing_prerequisite>\n\n"
        f"<active_remedies_task_level_deadlock_pattern count={len(task_level_deadlock)}>\n"
        f"{json.dumps(task_level_deadlock, ensure_ascii=False, indent=2)}\n"
        f"</active_remedies_task_level_deadlock_pattern>\n\n"
        f"<active_remedies_task_level_recovery_strategy count={len(task_level_recovery)}>\n"
        f"{json.dumps(task_level_recovery, ensure_ascii=False, indent=2)}\n"
        f"</active_remedies_task_level_recovery_strategy>\n\n"
        "Produce the repaired plan JSON now. Output JSON ONLY."
    )
