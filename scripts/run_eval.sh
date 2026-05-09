#!/usr/bin/env bash
# Run a single benchmark group with a given LLM backend.
#
# Usage:
#   scripts/run_eval.sh                              # default: wooden + qwen_plus
#   scripts/run_eval.sh iron qwen_plus               # iron tier with qwen plus
#   scripts/run_eval.sh diamond gpt_5_5              # diamond tier with gpt
#
# The server (scripts/server.sh) MUST already be running on
# ${MINEEVOLVE_PORT:-9000} on the same host (override via server.url=...).
set -euo pipefail

BENCHMARK="${1:-wooden}"
LLM="${2:-qwen_plus}"

PORT="${MINEEVOLVE_PORT:-9000}"

python -m mineevolve.main \
  benchmark="${BENCHMARK}" \
  llm="${LLM}" \
  server.port="${PORT}"
