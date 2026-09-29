#!/usr/bin/env bash
# Re-grade every finished conversation with every model, one judge at a time.
#
# The original runs graded each conversation with the model that produced it, so
# self-preference and judge leniency are inseparable in those numbers. This fills
# in the whole judge x assistant matrix -- every model judges every model's logs,
# itself included -- which is what makes the diagonal interpretable.
#
# One judge is served at a time; while it is up it grades all three runs and all
# assistants, then it is unloaded and the next is downloaded and served. Verdicts
# are cached per (run, judge) and outputs are append-only and keyed by conv_id,
# so the script is resumable: rerunning skips everything already graded.
#
# Grading is TEXT ONLY. The checkpoints are VLMs, but the judge reads a rendered
# transcript in which every image part is a bare `[image]` -- no pixels and no
# image paths reach it. That is why this serves a much shorter context than
# run_all_open_models.sh and passes no --limit-mm-per-prompt: without images a
# transcript is a few thousand tokens, and the spare memory becomes KV cache.
#
# Usage:
#   bash run_posthoc_judging.sh                     # all judges, all runs
#   bash run_posthoc_judging.sh --judges qwen3.5-9b # one judge
#   bash run_posthoc_judging.sh --smoke             # 5 conversations per file
#
# Sharding across GPUs: give each launch one run folder and one GPU. Each shard
# still cycles through every judge, so the judge x assistant matrix comes out
# complete for the run it owns. PORT defaults to 8301+GPU_ID, and the server
# logs, report and CSV directory are all namespaced by the run, so nothing
# collides. DOWNLOAD=0 is required: `hf download` writes lockfiles into a shared
# MODELS_DIR and concurrent shards would race on them.
#
#   GPU_ID=0 DOWNLOAD=0 RUNS="logs/Final-0" bash run_posthoc_judging.sh &
#   GPU_ID=1 DOWNLOAD=0 RUNS="logs/Final-1" bash run_posthoc_judging.sh &
#   GPU_ID=2 DOWNLOAD=0 RUNS="logs/Final-2" bash run_posthoc_judging.sh &
#
# Optional environment overrides:
#   GPU_ID=0 PORT=8301 MODELS_DIR=/path/to/models N_WORKERS=8 DOWNLOAD=1 \
#   RUNS="logs/Final-0 logs/Final-1 logs/Final-2" bash ...
#   TARGET_MAX_MODEL_LEN=32768 bash ...

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

JUDGE_FILTER=""
ASSISTANT_FILTER=""
SMOKE_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --judges)
      JUDGE_FILTER="${2:-}"
      [[ -z "$JUDGE_FILTER" ]] && { echo "--judges needs a value" >&2; exit 2; }
      shift 2
      ;;
    --assistants)
      # Whose logs get judged. Default is every model in MODELS, which is what
      # fills the matrix; narrow it when only some models have new logs, so a
      # pass does not re-walk 200k already-graded conversations to skip them.
      ASSISTANT_FILTER="${2:-}"
      [[ -z "$ASSISTANT_FILTER" ]] && { echo "--assistants needs a value" >&2; exit 2; }
      shift 2
      ;;
    --smoke)
      SMOKE_ARGS=(--limit 5)
      shift
      ;;
    -h|--help)
      sed -n '2,39p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash $0 [--judges a,b] [--assistants a,b] [--smoke]" >&2
      exit 2
      ;;
  esac
done

VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/../../.venv}"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Virtual environment not found: $VENV_DIR/bin/activate" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

GPU_ID="${GPU_ID:-0}"

export VLLM_PLUGINS=""
# Derived from GPU_ID so three shards launched on three GPUs cannot silently
# land on the same endpoint -- the second would abort on the "port already has
# a server" check, and worse, a stale server there would grade every shard with
# whatever model it happens to be holding. Base is above run_all_open_models.sh
# so a judging pass and a simulation pass can share a machine.
# Derived from the FIRST device: with GPU_ID=0,1 the whole string would go into
# the arithmetic, where bash reads the comma as an operator and yields 8302 --
# silently colliding with a GPU_ID=1 launch.
export PORT="${PORT:-$((8301 + ${GPU_ID%%,*}))}"
export VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"

N_WORKERS="${N_WORKERS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
# Text-only grading: a transcript with no images is a few thousand tokens, so
# 32k is generous. Overridable for a checkpoint that declares less.
TARGET_MAX_MODEL_LEN="${TARGET_MAX_MODEL_LEN:-32768}"
MIN_MODEL_LEN="${MIN_MODEL_LEN:-}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-1800}"
MODELS_DIR="${MODELS_DIR:-$SCRIPT_DIR/../../models}"
RUNS="${RUNS:-logs/Final-0 logs/Final-1 logs/Final-2}"

# `hf download` writes lockfiles and partial blobs inside MODELS_DIR. Several
# shards sharing that directory would race on them, so a parallel launch over
# already-present checkpoints must pass DOWNLOAD=0. It is not the default:
# skipping the fetch silently would turn a missing model into a confusing vLLM
# load failure instead of an obvious download.
DOWNLOAD="${DOWNLOAD:-1}"

# Every per-launch output path is namespaced by the runs this launch owns, so
# parallel shards never write the same file. Without it three processes would
# truncate the same server log and overwrite the same report.
RUN_TAG="$(for r in $RUNS; do basename "$r"; done | paste -sd- -)"

# alias|HF repo|launch profile|context ceiling|context floor|TP
#
# Every model in this list judges every model in it, itself included, so the
# list is both the judge set and the default assistant set -- keeping them one
# list is what guarantees the matrix stays square. Ordered by parameter count.
#
# An entry belongs here once that model has conversation logs to be judged.
# Uncomment the 30B/32B pair after their Experiment-A/B runs land; until then
# they are neither assistants nor judges, and the matrix is 9x9.
#
# TP is per entry for the same reason as in run_all_open_models.sh: the two
# large checkpoints are ~60-64GB of bf16 weights and cannot fit one 64GB card,
# while TP>1 on a small one idles a GPU. GPU_ID must name at least TP devices.
MODELS=(
  "qwen2.5-vl-3b|Qwen/Qwen2.5-VL-3B-Instruct|standard|32768|16384|1"
  "qwen3-vl-4b|Qwen/Qwen3-VL-4B-Instruct|standard|32768|16384|1"
  "qwen2.5-vl-7b|Qwen/Qwen2.5-VL-7B-Instruct|standard|32768|16384|1"
  "qwen3-vl-8b|Qwen/Qwen3-VL-8B-Instruct|standard|32768|16384|1"
  "internvl3.5-8b|OpenGVLab/InternVL3_5-8B|remote-code|32768|16384|1"
  "ministral3-8b-instruct|mistralai/Ministral-3-8B-Instruct-2512-BF16|mistral|32768|16384|1"
  "qwen3.5-9b|Qwen/Qwen3.5-9B|qwen|32768|16384|1"
  "pixtral-12b|mistralai/Pixtral-12B-2409|mistral|32768|16384|1"
  "qwen3.6-27b|Qwen/Qwen3.6-27B|qwen|32768|16384|1"
  "qwen3-vl-30b-a3b|Qwen/Qwen3-VL-30B-A3B-Instruct|standard|32768|16384|2"
  "qwen2.5-vl-32b|Qwen/Qwen2.5-VL-32B-Instruct|standard|32768|16384|2"
)

for command_name in python hf vllm curl setsid; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command not found after activating $VENV_DIR: $command_name" >&2
    exit 1
  fi
done

for run in $RUNS; do
  if [[ ! -d "$run/layered_vqa" ]]; then
    echo "No conversation logs under $run/layered_vqa" >&2
    exit 1
  fi
done

SERVER_LOG_DIR="${SERVER_LOG_DIR:-logs/posthoc_vllm_servers/$RUN_TAG}"
mkdir -p "$MODELS_DIR" "$SERVER_LOG_DIR"

# The assistant set is the judge set: every model grades every model, itself
# included. Derived from MODELS so the two cannot drift apart, unless
# --assistants narrows it for this launch.
ASSISTANTS=()
if [[ -n "$ASSISTANT_FILTER" ]]; then
  IFS=',' read -ra ASSISTANTS <<< "$ASSISTANT_FILTER"
else
  for spec in "${MODELS[@]}"; do
    ASSISTANTS+=("${spec%%|*}")
  done
fi
echo "Judging assistants: ${ASSISTANTS[*]}"

VLLM_PID=""

stop_vllm() {
  if [[ -z "$VLLM_PID" ]]; then
    return
  fi
  if kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "Stopping vLLM process group $VLLM_PID ..."
    kill -TERM -- "-$VLLM_PID" 2>/dev/null || kill -TERM "$VLLM_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 "$VLLM_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "vLLM did not stop after 60 seconds; killing its process group." >&2
      kill -KILL -- "-$VLLM_PID" 2>/dev/null || kill -KILL "$VLLM_PID" 2>/dev/null || true
    fi
  fi
  wait "$VLLM_PID" 2>/dev/null || true
  VLLM_PID=""
}

trap stop_vllm EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_server() {
  local served="$1" server_log="$2" waited=0
  while (( waited < SERVER_WAIT_SECONDS )); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "vLLM exited while loading $served. Last 100 log lines:" >&2
      tail -n 100 "$server_log" >&2 || true
      return 1
    fi
    if response="$(curl -fsS --max-time 5 "$VLLM_BASE_URL/models" 2>/dev/null)" && \
       python -c 'import json,sys; t=sys.argv[1]; d=json.load(sys.stdin); raise SystemExit(0 if any(x.get("id")==t for x in d.get("data",[])) else 1)' \
         "$served" <<<"$response"; then
      echo "vLLM is ready: $served"
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done
  echo "Timed out after ${SERVER_WAIT_SECONDS}s waiting for $served. Last 100 log lines:" >&2
  tail -n 100 "$server_log" >&2 || true
  return 1
}

resolve_max_model_len() {
  # min(entry ceiling, TARGET_MAX_MODEL_LEN, whatever the checkpoint declares).
  # Reading the config beats hardcoding: an entry that cannot reach its floor
  # fails here rather than dropping its longest transcripts one 400 at a time.
  local model_path="$1" ceiling="$2" alias="$3" floor="$4" resolved
  [[ -n "$MIN_MODEL_LEN" ]] && floor="$MIN_MODEL_LEN"

  resolved="$(python - "$model_path" "$ceiling" "$TARGET_MAX_MODEL_LEN" <<'PY'
import json, sys
from pathlib import Path

model_path, ceiling, target = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
KEYS = ("max_position_embeddings", "model_max_length", "max_seq_len")
NESTED = ("text_config", "llm_config", "language_config", "llm_cfg")


def windows(config):
    for key in KEYS:
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            yield value
    for key in NESTED:
        nested = config.get(key)
        if isinstance(nested, dict):
            yield from windows(nested)


found = []
for name in ("config.json", "params.json"):
    path = model_path / name
    if path.exists():
        try:
            found.extend(windows(json.loads(path.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, OSError):
            pass

limits = [ceiling, target]
if found:
    limits.append(max(found))
else:
    print("could not read a context window from the checkpoint config; "
          "using the requested target", file=sys.stderr)
print(min(limits))
PY
)" || return 1

  if (( resolved < floor )); then
    echo "$alias resolves to a ${resolved}-token context, below its ${floor}-token floor." >&2
    echo "Re-run with MIN_MODEL_LEN=$resolved if that is acceptable." >&2
    return 1
  fi
  echo "$resolved"
}

resolve_served_name() {
  # Serve under whatever models.yaml routes this alias to, since that is what
  # --judge_model puts on the wire. Also catches a missing entry now rather
  # than after a 20-minute load.
  python - "$1" <<'PY'
import sys
from pathlib import Path

import yaml

alias = sys.argv[1]
config = yaml.safe_load(Path("models.yaml").read_text(encoding="utf-8"))
entry = (config.get("models") or {}).get(alias)
if entry is None:
    raise SystemExit(f"{alias} has no models.yaml entry; --judge_model {alias} would not route")
print(entry.get("model", alias))
PY
}

assert_chat_template() {
  # A checkpoint with no usable template serves fine and then fails every
  # request, which downstream reads as a judge that graded everything wrong.
  python - "$1" "$2" "$3" <<'PY'
import json, sys
from pathlib import Path

model_path, profile, alias = Path(sys.argv[1]), sys.argv[2], sys.argv[3]

if profile == "mistral":
    found = [n for n in ("tekken.json", "tokenizer.model.v7m1", "tokenizer.model.v7",
                         "tokenizer.model.v3", "tokenizer.model")
             if (model_path / n).exists()]
    if not found:
        raise SystemExit(f"{alias}: mistral profile needs a Mistral tokenizer in {model_path}")
    print(f"{alias}: mistral-format tokenizer ({found[0]})")
    raise SystemExit(0)

for name in ("chat_template.jinja", "chat_template.json"):
    if (model_path / name).exists():
        print(f"{alias}: {name}")
        break
else:
    config_path = model_path / "tokenizer_config.json"
    template = None
    if config_path.exists():
        try:
            template = json.loads(config_path.read_text(encoding="utf-8")).get("chat_template")
        except (json.JSONDecodeError, OSError):
            template = None
    if not template:
        raise SystemExit(
            f"{alias}: no chat template in {model_path}. vLLM would serve this "
            f"checkpoint and then fail every grading request."
        )
    print(f"{alias}: chat_template in tokenizer_config.json")
PY
}

start_vllm() {
  local alias="$1" profile="$2" ceiling="$3" floor="$4" model_path="$5" tp="${6:-1}"
  local server_log="$SERVER_LOG_DIR/${alias}.log"
  local served max_len
  local -a args

  # GPU_ID becomes CUDA_VISIBLE_DEVICES, so it bounds the tensor-parallel size.
  # Refusing here turns "TP=2 on one GPU" into one line rather than a vLLM
  # stack trace several minutes into loading.
  local -a visible_gpus
  IFS=',' read -ra visible_gpus <<< "$GPU_ID"
  if (( tp > ${#visible_gpus[@]} )); then
    echo "$alias needs tensor-parallel-size $tp but GPU_ID names only ${#visible_gpus[@]} device(s): $GPU_ID" >&2
    echo "Re-run that judge with e.g. GPU_ID=0,1" >&2
    return 1
  fi

  max_len="$(resolve_max_model_len "$model_path" "$ceiling" "$alias" "$floor")" || return 1
  echo "Context length for $alias: $max_len (text-only grading)"
  echo "Chat template: $(assert_chat_template "$model_path" "$profile" "$alias")" || return 1
  served="$(resolve_served_name "$alias")" || return 1

  args=(
    "$model_path"
    --served-model-name "$served"
    --host 127.0.0.1
    --port "$PORT"
    --tensor-parallel-size "$tp"
    --max-model-len "$max_len"
    --max-num-seqs "$N_WORKERS"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --generation-config vllm
  )

  case "$profile" in
    qwen)        args+=(--reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}') ;;
    mistral)     args+=(--tokenizer_mode mistral --config_format mistral --load_format mistral) ;;
    remote-code) args+=(--trust-remote-code --default-chat-template-kwargs '{"enable_thinking": false}') ;;
    standard)    ;;
    *) echo "Unknown vLLM launch profile: $profile" >&2; return 1 ;;
  esac

  if curl -fsS --max-time 3 "$VLLM_BASE_URL/models" >/dev/null 2>&1; then
    echo "Port $PORT already has an OpenAI-compatible server. Stop it first." >&2
    return 1
  fi

  echo "Serving $alias as judge; server log: $server_log"
  setsid env HIP_VISIBLE_DEVICES="$GPU_ID" CUDA_VISIBLE_DEVICES="$GPU_ID" \
    vllm serve "${args[@]}" >"$server_log" 2>&1 &
  VLLM_PID=$!
  wait_for_server "$served" "$server_log"
}

echo "Model directory: $(cd "$MODELS_DIR" && pwd)"
echo "GPU: $GPU_ID"
echo "vLLM endpoint: $VLLM_BASE_URL"
echo "Runs to grade: $RUNS"
echo "vLLM version: $(vllm --version 2>&1 | head -n 1)"

for spec in "${MODELS[@]}"; do
  IFS='|' read -r alias hf_id profile ceiling floor tp <<<"$spec"
  tp="${tp:-1}"

  if [[ -n "$JUDGE_FILTER" && ",$JUDGE_FILTER," != *",$alias,"* ]]; then
    continue
  fi

  echo
  echo "============================================================================="
  echo "JUDGE: $alias  ($hf_id)"
  echo "============================================================================="

  model_path="$MODELS_DIR/$alias"
  # Resumable by design: `hf download` verifies what is already on disk and
  # fetches only what is missing, so a model left over from the simulation runs
  # costs one manifest check rather than a re-download.
  if [[ "$DOWNLOAD" == "1" ]]; then
    echo "Downloading/resuming $hf_id -> $model_path ..."
    hf download "$hf_id" --local-dir "$model_path"
  elif [[ -d "$model_path" ]]; then
    echo "Using existing checkpoint (DOWNLOAD=0): $model_path"
  else
    echo "DOWNLOAD=0 but $model_path does not exist." >&2
    echo "Fetch it once with DOWNLOAD=1, or point MODELS_DIR at the right place." >&2
    exit 1
  fi

  start_vllm "$alias" "$profile" "$ceiling" "$floor" "$model_path" "$tp"

  for run in $RUNS; do
    echo
    echo "--- [$alias] grading $run"
    # --assistants restricts grading to the six models that completed all three
    # runs. Final-0 also holds a partial internvl3-8b attempt; re-grading logs
    # that were never finished would put an incomparable row into the matrix.
    python posthoc_judge.py \
      --log_folder "$run" \
      --judge_model "$alias" \
      --assistants "${ASSISTANTS[@]}" \
      --workers "$N_WORKERS" \
      ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"}
  done

  stop_vllm
done

echo
echo "Post-hoc judging complete."
for run in $RUNS; do
  echo "  $run/posthoc/<judge>/<conv_type>/*.jsonl"
done

echo
echo "Aggregating ..."
# Namespaced by RUN_TAG: a shard covering one run writes its own report instead
# of clobbering a sibling's. Once every shard has finished, run the analysis
# again over all three at once to get the cross-run and cross-judge numbers:
#   python posthoc_analysis.py --runs logs/Final-0 logs/Final-1 logs/Final-2
python posthoc_analysis.py --runs $RUNS \
  --out "logs/posthoc_report_${RUN_TAG}.md" \
  --csv_dir "logs/posthoc_csv_${RUN_TAG}"
