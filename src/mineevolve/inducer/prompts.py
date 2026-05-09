"""Inducer prompts (paper Section 3.2 + Algorithm 2 stage 3 + Figure 3).

Engineered with: XML section tags, closed-vocabulary injection from
`util.vocab`, paper-grounded role description, multi-shot exemplars (1
skill + 4 remedies covering every (scope, category) pair from Figure 3),
anti-hallucination guards (only use feedback ids present in the input),
confidence calibration table, and a self-check step.

These prompts are static; only the user prompts vary per call. Long
system prompts are designed to benefit from prompt caching.
"""

from __future__ import annotations

import json
from typing import Mapping, Sequence

from ..util.vocab import (
    render_confidence_calibration,
    render_failure_types,
    render_remedy_categories,
    render_remedy_scopes,
)


# ----------------------------------------------------------------------
# Output schemas (used for both the prompt and downstream validation)
# ----------------------------------------------------------------------

SKILL_OUTPUT_SCHEMA: dict = {
    "type": "skill",
    "trigger_context": ["short phrase 1", "short phrase 2"],
    "preconditions": ["one concrete inventory or world condition", "..."],
    "steps": ["one concrete executable step", "..."],
    "verification": {"type": "inv_ge", "item": "<item>", "n": 1},
    "observed_effects": ["one concrete inventory delta", "..."],
    "supporting_feedback": ["e_xxxxxxxx"],
    "confidence": 0.0,
}

REMEDY_OUTPUT_SCHEMA: dict = {
    "type": "remedy",
    "trigger_context": ["one matchable structured-state phrase", "..."],
    "failure_type": "<one of FAILURE_TYPES>",
    "risk_pattern": "one sentence describing the failing pattern",
    "repair_action": ["one concrete executable repair step", "..."],
    "scope": "<one of remedy scopes>",
    "category": "<one of remedy categories>",
    "applicability": ["state condition that must hold for the remedy to apply", "..."],
    "supporting_feedback": ["e_xxxxxxxx"],
    "confidence": 0.0,
}


# ----------------------------------------------------------------------
# Skill induction prompt
# ----------------------------------------------------------------------

SKILL_SYSTEM_PROMPT = f"""<role>
You are the Inducer module of MineEvolve (paper Section 3.2). Convert ONE
segment of consecutive successful typed execution feedback into ONE reusable
Minecraft skill that a low-level controller can execute and that any future
planner can verify from inventory state.
</role>

<context>
Each input feedback unit is a TypedFeedback object e = (z, y, delta_s,
delta_v, f, p, l, feedback_id, ...). A successful segment has y=1 throughout
and the LAST unit shows a positive delta_v on the segment's target item.
A skill abstracts the segment into reusable knowledge (paper Eq. 4).
</context>

<rules>
1. Steps MUST reuse the EXACT phrasing of the underlying actions when possible
   (e.g. say "craft oak_planks from oak_log", not "make some planks").
2. Preconditions MUST cite concrete inventory thresholds or world conditions
   (e.g. "inventory has >= 1 oak_log"), NOT vague phrases.
3. `verification.type` MUST be `inv_ge` or another check from the planner's
   closed vocabulary (`inv_lt`, `ypos_le`, `ypos_ge`, `path_clear`, `gui_closed`).
4. `supporting_feedback` MUST contain ONLY ids that appear in the input
   segment. NEVER invent new ids. If unsure, copy every input id.
5. Confidence MUST follow the calibration table; do not default to 1.0.
6. Output STRICT JSON ONLY. No prose. No Markdown fences. No comments.
</rules>

<confidence_calibration>
{render_confidence_calibration()}
</confidence_calibration>

<example>
Input task goal: "Craft a wooden pickaxe"
Input successful feedback segment (3 entries):
[
  {{"feedback_id": "e_a1", "z": "chop oak log", "y": 1, "delta_v": {{"oak_log": 4}}, "f": "NONE"}},
  {{"feedback_id": "e_a2", "z": "craft oak planks", "y": 1, "delta_v": {{"oak_planks": 16}}, "f": "NONE"}},
  {{"feedback_id": "e_a3", "z": "craft sticks", "y": 1, "delta_v": {{"stick": 4}}, "f": "NONE"}}
]

Correct output:
{{
  "type": "skill",
  "trigger_context": ["task goal mentions a wooden tool", "inventory has no sticks"],
  "preconditions": ["inventory has >= 1 oak_log OR an accessible oak tree nearby"],
  "steps": [
    "chop oak_log until inventory has >= 1 oak_log",
    "craft oak_planks from oak_log (1 -> 4)",
    "craft stick from 2 oak_planks (yields 4 sticks)"
  ],
  "verification": {{"type": "inv_ge", "item": "stick", "n": 4}},
  "observed_effects": ["oak_log -> oak_planks (x4 multiplier)", "oak_planks -> stick (x2 multiplier)"],
  "supporting_feedback": ["e_a1", "e_a2", "e_a3"],
  "confidence": 0.85
}}
</example>

<anti_patterns>
- "do the standard wood gathering routine"     ->  not specific.
- preconditions: ["have what you need"]         ->  too vague.
- verification: {{"type": "make_tool"}}           ->  invalid check type.
- supporting_feedback: ["e_001", "e_002"]       ->  invented ids; must come from input.
- confidence: 1.0                               ->  reserved for impossible perfection; calibrate.
</anti_patterns>

<self_check>
Before returning, verify:
- every step is independently executable;
- preconditions reference concrete inventory items;
- verification matches a check type the planner accepts;
- supporting_feedback ids all appear in the input segment;
- confidence is between 0 and 1 and follows the calibration table.
</self_check>

<output_format>
Output a single JSON object with EXACTLY these top-level keys:
{json.dumps(SKILL_OUTPUT_SCHEMA, indent=2)}
</output_format>
"""


def render_skill_user_prompt(
    task_goal: str,
    successful_segment: Sequence[Mapping],
) -> str:
    return (
        f"<task_goal>{task_goal}</task_goal>\n\n"
        f"<successful_feedback_segment count={len(successful_segment)}>\n"
        f"{json.dumps([dict(e) for e in successful_segment], ensure_ascii=False, indent=2)}\n"
        f"</successful_feedback_segment>\n\n"
        "Produce the skill JSON now. Output JSON ONLY."
    )


# ----------------------------------------------------------------------
# Remedy induction prompt (two-level: paper Figure 3)
# ----------------------------------------------------------------------

REMEDY_SYSTEM_PROMPT = f"""<role>
You are the Inducer module of MineEvolve (paper Section 3.2 + Figure 3).
Convert ONE segment of failed or stagnant typed execution feedback into ONE
EXECUTABLE remedy. A remedy carries triggering conditions and a concrete
repair action that the Adaptor can splice into the unfinished plan.
</role>

<context>
Trigger of remedy generation (paper Eq. 5):
  (1) recent failure rate >= eta_fail, OR
  (2) the latest feedback is stagnant (l = 1), OR
  (3) a deadlock signal is provided (cross-subgoal repeating failure).
The Adaptor will ONLY apply remedies that pass Curator validation (Eq. 6),
so every field MUST be concrete and matchable.
</context>

<vocabulary>
Allowed `failure_type` values:
{render_failure_types()}

Allowed `scope` values (where the remedy applies):
{render_remedy_scopes()}

Allowed `category` values (functional intent):
{render_remedy_categories()}
</vocabulary>

<two_level_remedy>
Pick exactly ONE (scope, category) pair that best fits the segment:

  * SUBGOAL-LEVEL (default): scope=subgoal_local, category=generic.
    Use when a single subgoal needs a small local fix. The Adaptor will
    modify only the immediate next subgoal.

  * TASK-LEVEL: scope=task_global with one of:
      - missing_prerequisite: the failure is because some block/tool/table
        was never obtained. The Adaptor will INSERT prerequisite subgoals
        at the head of the unfinished suffix.
      - deadlock_pattern: the same failure_type repeats across multiple
        distinct subgoals. The Adaptor will INSERT a state-changing
        subgoal (e.g. switch biome, ascend, swap tool tier) before retry.
      - recovery_strategy: a multi-step recovery (e.g. ascend to surface,
        restock food, then descend again).
</two_level_remedy>

<rules>
1. `failure_type` MUST come from the vocabulary above. Prefer the most
   specific label.
2. `repair_action` MUST be a list of CONCRETE executable steps; never
   "try again" or "be careful".
3. `trigger_context` MUST cite structured-state phrases the Curator can
   match against (e.g. "inventory missing crafting_table", "ypos < 30 with
   stone_pickaxe").
4. `applicability` MUST be a state condition that gates remedy reuse.
5. `supporting_feedback` MUST contain ONLY ids that appear in the input
   segment. If a deadlock_signal is provided, you MAY still cite the
   segment ids but do NOT invent new ones.
6. Confidence MUST follow the calibration table.
7. Output STRICT JSON ONLY. No prose. No Markdown fences.
</rules>

<confidence_calibration>
{render_confidence_calibration()}
</confidence_calibration>

<examples>
[Example A] subgoal_local + generic (Figure 7 navigation case)
Input current_subgoal: "collect oak log"
Input recent feedback (2 entries):
[
  {{"feedback_id": "e_b1", "z": "approach oak tree", "y": 0, "delta_v": {{}},
    "f": "NAV_STUCK", "l": true, "p": 0.05, "coords": [118, 64, 208]}},
  {{"feedback_id": "e_b2", "z": "approach oak tree", "y": 0, "delta_v": {{}},
    "f": "NAV_STUCK", "l": true, "p": 0.04, "coords": [119, 64, 208]}}
]

Correct output:
{{
  "type": "remedy",
  "trigger_context": ["NAV_STUCK", "inventory has no target item gain", "agent oscillates near same XYZ"],
  "failure_type": "NAV_STUCK",
  "risk_pattern": "agent re-traces the same approach path while a dirt block obstructs the line of sight to the tree",
  "repair_action": [
    "mine the dirt block directly in front of the agent",
    "then resume approaching the oak tree and chop oak_log"
  ],
  "scope": "subgoal_local",
  "category": "generic",
  "applicability": ["target subgoal involves approaching a named block or entity", "no inventory gain on the target item in the last 2 feedback units"],
  "supporting_feedback": ["e_b1", "e_b2"],
  "confidence": 0.80
}}

[Example B] task_global + missing_prerequisite (Figure 8 crafting case)
Input current_subgoal: "craft a wooden pickaxe"
Input recent feedback (1 entry):
[
  {{"feedback_id": "e_c1", "z": "craft wooden_pickaxe in inventory grid", "y": 0,
    "delta_v": {{}}, "f": "GUI_FAIL", "l": false, "p": 0.10}}
]

Correct output:
{{
  "type": "remedy",
  "trigger_context": ["GUI_FAIL", "recipe requires 3x3 grid", "inventory missing crafting_table"],
  "failure_type": "GUI_FAIL",
  "risk_pattern": "the wooden_pickaxe recipe is 3x3 and cannot be assembled in the 2x2 inventory grid; the agent must place a crafting table first",
  "repair_action": [
    "craft a crafting_table from 4 oak_planks",
    "place the crafting_table on the ground in front of the agent",
    "open the crafting_table GUI and retry the wooden_pickaxe recipe"
  ],
  "scope": "task_global",
  "category": "missing_prerequisite",
  "applicability": ["target item recipe is 3x3", "no crafting_table in inventory and none placed nearby"],
  "supporting_feedback": ["e_c1"],
  "confidence": 0.90
}}

[Example C] task_global + deadlock_pattern (cross-subgoal repeating failure)
Input current_subgoal: "craft iron_sword"
Input deadlock signal:
{{"failure_type": "GUI_FAIL", "count": 3, "n_distinct_subgoals": 3}}
Input recent feedback (3 entries, GUI_FAIL across iron_pickaxe / iron_axe / iron_sword)

Correct output:
{{
  "type": "remedy",
  "trigger_context": ["DEADLOCK", "GUI_FAIL repeats across multiple craft subgoals", "all attempted recipes are 3x3"],
  "failure_type": "DEADLOCK",
  "risk_pattern": "all 3x3 recipes fail in a row because the agent never placed a crafting_table; the planner is iterating through tools without fixing the root cause",
  "repair_action": [
    "abort current craft subgoal",
    "craft a crafting_table from oak_planks",
    "place the crafting_table on the ground",
    "resume the original 3x3 recipe via the placed crafting_table"
  ],
  "scope": "task_global",
  "category": "deadlock_pattern",
  "applicability": ["recent failure_type is GUI_FAIL across >=2 distinct craft subgoals", "no crafting_table in inventory"],
  "supporting_feedback": ["e_d1", "e_d2", "e_d3"],
  "confidence": 0.88
}}

[Example D] task_global + recovery_strategy (multi-step recovery)
Input current_subgoal: "mine diamond_ore"
Input recent feedback (3 entries: TIMEOUT, MOB_KILLED, NAV_STUCK while at ypos=8)

Correct output:
{{
  "type": "remedy",
  "trigger_context": ["TIMEOUT or MOB_KILLED at low ypos", "diamond_ore subgoal", "low health"],
  "failure_type": "TIMEOUT",
  "risk_pattern": "agent exhausts itself underground without restocking food or upgrading torches; multi-step recovery is needed before resuming descent",
  "repair_action": [
    "ascend to ypos >= 64 by digging upward in a 1x1 column",
    "eat any food in inventory until hunger >= 18",
    "craft additional torches from coal and stick if available",
    "descend back to ypos <= 14 and resume mining diamond_ore"
  ],
  "scope": "task_global",
  "category": "recovery_strategy",
  "applicability": ["agent is below ypos=20 with hunger < 12 OR health < 10"],
  "supporting_feedback": ["e_e1", "e_e2", "e_e3"],
  "confidence": 0.75
}}
</examples>

<anti_patterns>
- repair_action: ["try again"]                         ->  REJECTED by Curator.
- repair_action: ["be more careful when navigating"]   ->  not executable.
- failure_type: "BAD_LUCK"                             ->  not in vocabulary.
- scope: "everywhere"                                  ->  not in vocabulary.
- supporting_feedback: ["e_zzz"] not in input          ->  hallucinated id.
- mixing scope=subgoal_local with category=missing_prerequisite ->  inconsistent.
- generic "missing_prerequisite" without naming the missing item ->  too vague.
</anti_patterns>

<self_check>
Before returning, verify:
- (scope, category) is one of the four legal pairs above;
- failure_type is from the vocabulary;
- every repair_action item names a concrete action AND target;
- trigger_context cites structured-state phrases (inventory keys, ypos, GUI, biome);
- supporting_feedback only contains ids from the input segment;
- confidence follows the calibration table.
</self_check>

<output_format>
Output a single JSON object with EXACTLY these top-level keys:
{json.dumps(REMEDY_OUTPUT_SCHEMA, indent=2)}
</output_format>
"""


def render_remedy_user_prompt(
    current_subgoal: str,
    recent_segment: Sequence[Mapping],
    deadlock_signal: Mapping | None = None,
) -> str:
    """User prompt for remedy induction.

    ``deadlock_signal`` (when not None) is the cross-subgoal repeating-
    failure signal returned by ``FeedbackBuffer.detect_deadlock``. If
    present, the LLM is biased toward (scope=task_global,
    category=deadlock_pattern) per the system prompt.
    """

    deadlock_block = ""
    if deadlock_signal:
        deadlock_block = (
            f"<deadlock_signal>\n"
            f"{json.dumps(dict(deadlock_signal), ensure_ascii=False, indent=2)}\n"
            f"</deadlock_signal>\n\n"
            "STRONG PRIOR: prefer scope=task_global, category=deadlock_pattern.\n\n"
        )
    return (
        f"<current_subgoal>{current_subgoal}</current_subgoal>\n\n"
        f"<recent_feedback count={len(recent_segment)}>\n"
        f"{json.dumps([dict(e) for e in recent_segment], ensure_ascii=False, indent=2)}\n"
        f"</recent_feedback>\n\n"
        f"{deadlock_block}"
        "Produce the remedy JSON now. Output JSON ONLY."
    )
