#!/usr/bin/env bash
# Prove the closed-model pipeline works, on N questions instead of 300.
#
# This runs every delivery condition the real launch runs -- the 14 Experiment-B
# ones and the 3 Experiment-A ones, 17 in all (A1 is the same run as
# Experiment-B FULL, so it resumes rather than re-calling) -- against a dataset
# cut down to N samples. What it checks is plumbing: routing, the API key, the
# scripted and LLM user agents, image materialization, the judge, the resume
# accounting, and the scorer. It says nothing about accuracy; N samples cannot.
#
# The conditions come from lic_conditions.sh, the same file the real runner
# sources. A condition that only the smoke test knows about would be a
# condition the smoke test cannot vouch for.
#
# Usage:
#   bash smoke_test_closed_models.sh --n 1
#   bash smoke_test_closed_models.sh --n 1 --only gpt-5.6-luna
#   bash smoke_test_closed_models.sh --n 2 --only minimax-m3,gemini-3.7-flash
#   bash smoke_test_closed_models.sh --n 1 --dry-run    # print, call nothing
#   bash smoke_test_closed_models.sh --n 1 --no-posthoc # answers only
#
# The behavioural pass runs here too, so the smoke test covers it: it is the
# part most likely to break on a new model, since it needs the judge to return
# a structured verdict rather than a bare yes/no.
#
# Optional environment overrides:
#   N_WORKERS=4 LOG_FOLDER=logs/Smoke-Closed bash ...
#   KEEP_GOING=0 bash ...    # stop at the first failing condition
#   SKIP_VENV=1 bash ...     # use the python already on PATH
#
# THIS SPENDS MONEY, just very little of it: roughly N x (a few dozen calls) per
# model. --dry-run makes none.
#
# Failures do not abort by default: the point of a smoke test is to learn which
# conditions are broken, not just the first one, so every condition runs and the
# script exits non-zero with a PASS/FAIL line per condition.
#
# The cut-down datasets are written by make_smoke_datasets.py into
# ../../data/derived/lic/.smoke-n<N>/, keeping their original basenames because
# the logs and the scorers key conditions off those. Smoke logs go to their own
# LOG_FOLDER with their own judge cache, so they never mix with a real run's.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

N_SAMPLES=""
MODEL_FILTER=""
DRY_RUN=0
SKIP_POSTHOC="${SKIP_POSTHOC:-0}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --n|-n)
      N_SAMPLES="${2:-}"
      [[ -z "$N_SAMPLES" ]] && { echo "--n needs a value" >&2; exit 2; }
      shift 2
      ;;
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
      sed -n '2,39p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash $0 --n N [--only alias[,alias...]] [--dry-run] [--no-posthoc] [--log_folder DIR]" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$N_SAMPLES" ]]; then
  echo "--n is required: how many questions each condition runs." >&2
  echo "Usage: bash $0 --n N [--only alias[,alias...]] [--dry-run] [--no-posthoc]" >&2
  exit 2
fi
if ! [[ "$N_SAMPLES" =~ ^[0-9]+$ ]] || (( N_SAMPLES < 1 )); then
  echo "--n must be a positive integer, got: $N_SAMPLES" >&2
  exit 2
fi

VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/../../.venv}"
if [[ "${SKIP_VENV:-0}" == "1" ]]; then
  # For a --dry-run or a checkout without the venv built. A real launch wants
  # the venv, so this is opt-in rather than a silent fallback to system python.
  echo "SKIP_VENV=1: using whatever python is on PATH ($(command -v python || echo none))"
elif [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
elif [[ -f "$VENV_DIR/Scripts/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/Scripts/activate"
else
  echo "Virtual environment not found: $VENV_DIR" >&2
  exit 1
fi

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
if [[ -z "${OPENROUTER_API_KEY:-}" && "$DRY_RUN" != "1" ]]; then
  echo "OPENROUTER_API_KEY is not set and no value was found in $ENV_FILE" >&2
  exit 1
fi

SOURCE_DATASET_DIR="${SOURCE_DATASET_DIR:-../../data/derived/lic}"
SMOKE_DATASET_DIR="$SOURCE_DATASET_DIR/.smoke-n${N_SAMPLES}"

echo "Building $N_SAMPLES-sample datasets in $SMOKE_DATASET_DIR ..."
python make_smoke_datasets.py --n "$N_SAMPLES" --dataset_dir "$SOURCE_DATASET_DIR"

export N_WORKERS="${N_WORKERS:-2}"
export LOG_FOLDER="${LOG_FOLDER:-logs/Smoke-Closed-n${N_SAMPLES}}"
export LAYERED_VQA_JUDGE_CACHE="$SCRIPT_DIR/$LOG_FOLDER/judge_cache.jsonl"

# Report every broken condition in one pass, not just the first.
KEEP_GOING="${KEEP_GOING:-1}"
D="$SMOKE_DATASET_DIR"
export DRY_RUN KEEP_GOING SKIP_POSTHOC
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lic_conditions.sh"

mkdir -p "$LOG_FOLDER" "$(dirname "$LAYERED_VQA_JUDGE_CACHE")"
assert_datasets

# Same list as run_all_closed_models.sh, and --only picks a subset of it.
MODELS=(
  "gpt-5.6-luna"
  "minimax-m3"
  "gemini-3.7-flash"
)

assert_openrouter() {
  local alias="$1"
  python - "$alias" <<'PY'
import sys

from model_registry import registry

alias = sys.argv[1]
route = registry.resolve(alias)
if route.backend != "openrouter":
    raise SystemExit(
        f"{alias} routes to {route.backend}, not openrouter. This smoke test "
        f"covers the closed-model path; use run_all_open_models.sh for a "
        f"locally served checkpoint."
    )
PY
}

echo "Smoke datasets: $SMOKE_DATASET_DIR ($N_SAMPLES sample(s) each)"
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
  echo "SMOKE MODEL: $alias  (n=$N_SAMPLES)"
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
  echo "--only $MODEL_FILTER matched none of: ${MODELS[*]}" >&2
  exit 2
fi

print_condition_summary

failures="$(failed_condition_count)"
echo
if (( failures > 0 )); then
  echo "SMOKE TEST FAILED: $failures condition(s) did not complete."
  exit 1
fi
echo "SMOKE TEST PASSED: every condition completed on $N_SAMPLES sample(s)."
echo "Reports: $LOG_FOLDER/results_<model>.md"
