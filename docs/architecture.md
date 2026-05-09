# Architecture

MineEvolve is structured as a two-process system: an **env-side client** that owns the MineRL world and a **server** that hosts STEVE-1, the LLM planner, and the four MineEvolve modules. They communicate via FastAPI over HTTP.

## Data flow

```mermaid
flowchart TD
    subgraph ClientProc[Client process (CPU, MineRL env)]
        EnvLoop[Algorithm 1 main loop\nsrc/mineevolve/main.py]
        EnvLoop -->|reset / step / commands| MineRL[MineEvolveEnvWrapper\nsrc/mineevolve/env/wrapper.py]
        MineRL -->|inv coords gui pov| Status[StatusMod\nsrc/mineevolve/env/mods/status.py]
        Status --> EnvLoop
    end

    subgraph ServerProc[Server process (GPU, port 9000)]
        Plan[Planner LLM backend\nsrc/mineevolve/planner/backends/*]
        Plan --> Subgoals[Subgoal JSON\nsrc/mineevolve/planner/base.py]
        Subgoals --> Steve[SteveRunner\nsrc/mineevolve/executor/steve_runner.py]
        Monitor[Monitor: typed feedback\nsrc/mineevolve/monitor/*] --> Buffer[Feedback buffer B\nsrc/mineevolve/monitor/buffer.py]
        Buffer --> Inducer[Inducer: skill / remedy\nsrc/mineevolve/inducer/*]
        Inducer --> Curator[Curator: validate merge retrieve\nsrc/mineevolve/curator/*]
        Curator --> KB[(External KB JSON store)]
        Curator --> Adaptor[Adaptor: freeze prefix repair suffix\nsrc/mineevolve/adaptor/*]
        Adaptor --> Plan
    end

    EnvLoop -.->|/chat type=plan| Plan
    EnvLoop -.->|/chat type=action| Steve
    Steve -.->|action dict| EnvLoop
    EnvLoop -.->|/chat type=monitor| Monitor
    EnvLoop -.->|/chat type=induce| Inducer
    EnvLoop -.->|/chat type=repair| Adaptor
```

## Module map

| Layer        | Path                                  | Purpose                                                                                          |
| ------------ | ------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Env spec     | `env/custom_env.py`                   | MineRL `HumanSurvival` subclass for the 7 tech-tree env names                                    |
| Wrapper      | `env/wrapper.py`                      | Dynamic Y-band ore spawn, auto pickaxe, Y-dwell tracker (no environment-level stuck recovery)    |
| Inv. start   | `env/inventory_agent_start.py`        | Enforces `[]` initial inventory                                                                  |
| Mods         | `env/mods/{status,recorder,task_checker}.py` | State snapshot, optional video/action recording, inventory-delta success check            |
| Monitor      | `monitor/{feedback,progress,failure,buffer}.py` | Eq. 1 typed feedback, Eq. 2 progress score, Eq. 3 stagnation, episode buffer B                |
| Inducer      | `inducer/{schema,induce_skill,induce_remedy,prompts}.py` | Eq. 4 KnowledgeEntry, Eq. 5 trigger, Appendix E induction prompts                       |
| Curator      | `curator/{store,validation,retrieval,merge}.py` | JSON store, Eq. 6 validation rules, Eq. 7 budget retrieval, dedupe + confidence merge        |
| Adaptor      | `adaptor/{repair,prompts}.py`         | Eq. 8 freeze prefix + LLM-repair suffix, Adaptor prompts                                         |
| Planner      | `planner/{base,prompts,parser,backends/}.py` | Subgoal schema, Appendix E planner prompts, JSON parser, 5 OpenAI-compatible backends         |
| Executor     | `executor/{steve_loader,steve_runner,craft_helper}.py` | Loads STEVE-1 via MineStudio or the official `steve1` package; runs step-by-step       |
| Server       | `server/{api,routes,agent}.py` + `app.py` | FastAPI `/reset`, `/chat` (plan/action/monitor/induce/repair/advance/status)                |
| Client       | `client/server_api.py`                | `requests`-based wrapper used by the env-side process                                            |
| Main loop    | `main.py`                             | Hydra `@main` implementing Algorithm 1                                                           |
| Eval mons.   | `monitors/{success,step}.py` + `util/logger.py` | Per-task success rate, average step count, `rich.Table` results                            |

## Per-step contract on the server side

Each `/chat` request type maps to one method on `MineEvolveAgent`
(`server/agent.py`). All shared state (current plan, feedback buffer, KB)
lives there under a single `threading.Lock`. The client process is
stateless beyond the env it owns.

| `/chat` type | Method on agent           | Purpose                                                                 |
| ------------ | ------------------------- | ----------------------------------------------------------------------- |
| `plan`       | `plan(state, ...)`        | Initial planner call (paper Algorithm 1 line 6)                         |
| `action`     | `action(condition, obs)`  | Single STEVE-1 action step                                              |
| `monitor`    | `monitor_subgoal(...)`    | Push a TypedFeedback into B                                             |
| `induce`     | `induce_and_curate(...)`  | Run Inducer + Curator over the current B                                |
| `repair`     | `repair(state, ...)`      | Adaptor: freeze prefix, retrieve K, run repair prompt                   |
| `advance`    | `advance_subgoal(success)`| Move the cursor `i` in the active plan                                  |
| `status`     | `status()`                | Returns buffer size, KB stats, and current plan position                |

## Two-level remedies (paper Figure 3)

The paper's Figure 3 distinguishes two complementary remedy levels. We
encode the level on the `KnowledgeEntry.u` payload via `scope` (where the
remedy applies) and `category` (functional intent):

| Level         | scope                   | category               | Adaptor application                             |
| ------------- | ----------------------- | ---------------------- | ----------------------------------------------- |
| Subgoal-level | `subgoal_local`         | `generic`              | Modify ONLY the immediate next subgoal          |
| Task-level    | `task_global`           | `missing_prerequisite` | INSERT prerequisite subgoals at suffix HEAD     |
| Task-level    | `task_global`           | `deadlock_pattern`     | INSERT a state-changing subgoal at suffix HEAD  |
| Task-level    | `task_global`           | `recovery_strategy`    | INSERT a multi-step recovery sequence at HEAD   |

The Inducer prompt (`src/mineevolve/inducer/prompts.py`) explicitly asks
the LLM to pick one (scope, category) pair. The Adaptor prompt
(`src/mineevolve/adaptor/prompts.py`) buckets active remedies by category
so the LLM writes structurally correct repaired suffixes.

## Deadlock detection (Figure 3 task-level trigger)

`FeedbackBuffer.detect_deadlock(window, min_distinct_subgoals, min_repeats)`
returns a non-None signal when the SAME failure type repeats across at
least `min_distinct_subgoals` distinct subgoals within the recent window.
This is the cross-subgoal companion to Eq. (3)'s within-subgoal
stagnation flag and is what triggers `category=deadlock_pattern` remedy
generation.

When a deadlock is detected, the Inducer's remedy prompt receives a
`deadlock_signal` block that biases the LLM toward `scope=task_global` /
`category=deadlock_pattern`; if the LLM still returns `category=generic`,
`induce_remedy.build_remedy_via_llm` overrides it to the task-level
deadlock category.

## Failure type vocabulary (paper Figure 3 Monitor box)

`monitor.failure.FAILURE_TYPES` mirrors the four icons under "Failure /
Stagnation Signals" in Figure 3:

| Figure 3 label       | Vocabulary entry      | Diagnosis condition                                              |
| -------------------- | --------------------- | ---------------------------------------------------------------- |
| Navigation Stuck     | `NAV_STUCK`           | no inventory change AND coord span < `LOW_MOVEMENT_THRESHOLD`    |
| Target Unreachable   | `TARGET_UNREACHABLE`  | no inventory change AND coord span > `HIGH_MOVEMENT_THRESHOLD`   |
| GUI Blockage         | `GUI_FAIL`            | GUI open AND no inventory change                                 |
| Missing Tool         | `MISSING_ITEM`        | named target item AND no inventory gain on it                    |

Auxiliary entries: `RECIPE_FAIL`, `SMELT_FAIL`, `DEADLOCK`,
`MOB_KILLED`, `TIMEOUT`, `UNKNOWN`.

## Configuration

Hydra composes the active config from:

```
conf/evaluate.yaml          # base
conf/benchmark/<tier>.yaml  # 70-task subset, one file per tech-tree tier
conf/llm/<backend>.yaml     # one of qwen_flash | qwen_plus | glm_4_7 | gemini_flash | gpt_5_5
```

Override anything from the command line:

```bash
python -m mineevolve.main \
  benchmark=diamond \
  llm=glm_4_7 \
  runtime.budget_tokens=768 \
  runtime.eta_fail=0.4
```
