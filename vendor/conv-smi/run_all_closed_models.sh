#!/usr/bin/env bash
# Evaluate the closed-weight models sequentially through OpenRouter.
#
# The open-weight counterpart, run_all_open_models.sh, spends most of its length
# on the GPU: download the checkpoint, size the context window, serve it, wait,
# unload. None of that exists here. Every closed model is already served, so
# this script is the same 14 Experiment-B conditions plus Experiment A and the
# per-model report, run against an endpoint that bills per call.
#
# Both scripts get those conditions from lic_conditions.sh, so neither can drift
# from the other on what a condition is.
#
# Usage:
#   bash run_all_closed_models.sh                        # all three models
#   bash run_all_closed_models.sh --only gpt-5.6-luna    # one of them
#   bash run_all_closed_models.sh --dry-run              # print, call nothing
#   bash run_all_closed_models.sh --no-posthoc           # answers only
#   bash run_all_closed_models.sh --log_folder logs/Final-Closed-1 --only minimax-m3
#
# After each model finishes, its own logs are re-graded once more for the
# behavioural measures the simulators do not record -- which turn answered,
# what had been revealed by then, attempt counts, verbosity, citation. That is
# one extra judge call per conversation. --no-posthoc leaves it out.
#
# Optional environment overrides:
#   N_WORKERS=8 LOG_FOLDER=logs/Final-Closed-0 bash ...
#   OPENROUTER_API_KEY=sk-or-... bash ...   # otherwise read from ../../.env
#   SKIP_VENV=1 bash ...                    # use the python already on PATH
#
# THIS SPENDS MONEY. Every condition is one call per sample per turn against a
# metered API, and there is no cost accounting anywhere in this repo — check
# OpenRouter's dashboard, not the logs. --dry-run prints the exact commands
# without making a single call; run it first.
#
# The judge cache is pinned inside LOG_FOLDER, so a fresh LOG_FOLDER is a fresh
# set of verdicts rather than a replay of an earlier run's cached ones. Runs
# already logged in that folder are detected and skipped, so an interrupted
# launch resumes instead of paying twice.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_FILTER=""
DRY_RUN=0
SKIP_POSTHOC="${SKIP_POSTHOC:-0}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)
      MODEL_FILTER="${2:-}"
      [[ -z "$MODEL_FILTER" ]] && { echo "--only needs a value" >&2; exit 2; }
      shift 2
      ;;
    --log_folder|--log-folder)
      # Also settable as LOG_FOLDER=. Having only the env var was a trap: this
      # reads like every other entry point in the repo, and an unknown argument
      # aborts the launch rather than quietly running it in the default folder.
      LOG_FOLDER="${2:-}"
      [[ -z "$LOG_FOLDER" ]] && { echo "--log_folder needs a value" >&2; exit 2; }
      shift 2
      ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-posthoc) SKIP_POSTHOC=1; shift ;;
    -h|--help)
      sed -n '2,36p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash $0 [--only alias[,alias...]] [--dry-run] [--no-posthoc] [--log_folder DIR]" >&2
      exit 2
      ;;
  esac
done

VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/../../.venv}"
if [[ "${SKIP_VENV:-0}" == "1" ]]; then
  # For a --dry-run or a checkout without the venv built. A real launch wants
  # the venv, so this is opt-in rather than a silent fallback to system python.
  echo "SKIP_VENV=1: using whatever python is on PATH ($(command -v python || echo none))"
elif [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
elif [[ -f "$VENV_DIR/Scripts/activate" ]]; then
  # No GPU is involved here, so this one is runnable from a Windows checkout
  # too; the open-weight runner is Linux-only by necessity, this is not.
  # shellcheck disable=SC1091
  source "$VENV_DIR/Scripts/activate"
else
  echo "Virtual environment not found: $VENV_DIR" >&2
  exit 1
fi

# OPENROUTER_API_KEY: the environment wins; ../../.env is the fallback. Read
# with grep rather than by sourcing the file, so an unrelated line in .env
# cannot execute anything here.
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/../../.env}"
if [[ -z "${OPENROUTER_API_KEY:-}" && -f "$ENV_FILE" ]]; then
  key_line="$(grep -E '^[[:space:]]*(export[[:space:]]+)?OPENROUTER_API_KEY=' "$ENV_FILE" | tail -n 1 || true)"
  if [[ -n "$key_line" ]]; then
    key_value="${key_line#*=}"
    key_value="${key_value%$'\r'}"
    key_value="${key_value%\"}"; key_value="${key_value#\"}"
    key_value="${key_value%\'}"; key_value="${key_value#\'}"
    export OPENROUTER_API_KEY="$key_value"
  fi
fi
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is not set and no value was found in $ENV_FILE" >&2
  exit 1
fi

# No VLLM_BASE_URL is needed: nothing here routes to the vllm backend, and
# clients are built lazily per backend.
export N_WORKERS="${N_WORKERS:-4}"
export LOG_FOLDER="${LOG_FOLDER:-logs/Final-Closed-0}"
export LAYERED_VQA_JUDGE_CACHE="$SCRIPT_DIR/$LOG_FOLDER/judge_cache.jsonl"

D="${D:-../../data/derived/lic}"
export DRY_RUN SKIP_POSTHOC
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lic_conditions.sh"

mkdir -p "$LOG_FOLDER" "$(dirname "$LAYERED_VQA_JUDGE_CACHE")"
assert_datasets

# Aliases from models.yaml. Adding a model here means adding it there too --
# an unknown name is an error rather than a guess, which is the point.
MODELS=(
  "gpt-5.6-luna"
  "minimax-m3"
  "gemini-3.7-flash"
)

assert_openrouter() {
  # A closed-model launch that quietly resolved to the vllm backend would fail
  # on connection; an open-model alias that quietly resolved to OpenRouter
  # would bill an account. Check the routing before the first call, not after.
  local alias="$1"
  python - "$alias" <<'PY'
import sys

from model_registry import registry

alias = sys.argv[1]
route = registry.resolve(alias)
if route.backend != "openrouter":
    raise SystemExit(
        f"{alias} routes to {route.backend}, not openrouter. "
        f"run_all_closed_models.sh only runs metered API models; use "
        f"run_all_open_models.sh for locally served checkpoints."
    )
PY
}

echo "OpenRouter endpoint: https://openrouter.ai/api/v1"
echo "Log folder: $LOG_FOLDER"
echo "Judge cache: $LAYERED_VQA_JUDGE_CACHE"
echo "Workers: $N_WORKERS"
[[ "$DRY_RUN" == "1" ]] && echo "DRY RUN: no API calls will be made"

ran=0
for alias in "${MODELS[@]}"; do
  if [[ -n "$MODEL_FILTER" && ",$MODEL_FILTER," != *",$alias,"* ]]; then
    continue
  fi
  ran=$((ran + 1))

  echo
  echo "============================================================================="
  echo "MODEL: $alias"
  echo "============================================================================="

  assert_openrouter "$alias"
  if [[ "$DRY_RUN" != "1" ]]; then
    python agent_config.py --model "$alias"
  fi

  run_exp_b "$alias"
  run_exp_a "$alias"
  write_model_report "$alias"
  run_posthoc "$alias"
done

if (( ran == 0 )); then
  # Silently running nothing looks exactly like a finished run in a log.
  echo "--only $MODEL_FILTER matched none of: ${MODELS[*]}" >&2
  exit 2
fi

print_condition_summary

echo
echo "All requested closed-model runs are complete."
