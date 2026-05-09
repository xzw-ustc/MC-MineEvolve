# 70-task evaluation suite (paper Table 3)

The seven tech-tree groups below sum to **70 tasks**. Each group is defined in
`src/mineevolve/conf/benchmark/<group>.yaml` with `env.initial_inventory: []`
(no item is given to the agent at episode start).

| Group     | Count | Yaml file                                   | Episode horizon |
| --------- | ----: | ------------------------------------------- | --------------- |
| Wooden    |    11 | `conf/benchmark/wooden.yaml`                | 2 minutes       |
| Stone     |    10 | `conf/benchmark/stone.yaml`                 | 3 minutes       |
| Iron      |    16 | `conf/benchmark/iron.yaml`                  | 20 minutes      |
| Gold      |     7 | `conf/benchmark/gold.yaml`                  | 10 minutes      |
| Redstone  |     6 | `conf/benchmark/redstone.yaml`              | 15 minutes      |
| Diamond   |     7 | `conf/benchmark/diamond.yaml`               | 30 minutes      |
| Armor     |    13 | `conf/benchmark/armor.yaml`                 | 25 minutes      |
| **Total** |  **70** |                                             |                 |

## Easy / Hard split (paper Section 4.1)

- **Easy**: Wooden + Stone + Gold = 28 tasks
- **Hard**: Iron + Redstone + Diamond + Armor = 42 tasks

Aggregate scores in the paper (Easy Avg., Hard Avg., Overall) are computed
as task-count-weighted means; this repository's `util.logger.print_results`
prints the same per-task and overall metrics for the active benchmark
group. Combining results across groups for paper-style aggregates is left
to a downstream analysis script.

## Selecting a subset

Each yaml has `evaluate: []` which means "all tasks". Run a single id:

```bash
python -m mineevolve.main benchmark=iron benchmark.evaluate='[12]'
```

Or repeat each task `times` runs:

```bash
python -m mineevolve.main benchmark=diamond benchmark.env.times=5
```

## Adding a task

Edit the relevant `conf/benchmark/<group>.yaml`:

```yaml
all_task:
  - {id: 99, type: craft, instruction: "Craft a fishing rod"}
```

`type` is a label hint only (`craft`, `mine`, `smelt`, `kill`); the
planner generates the actual subgoal sequence from the instruction string.
