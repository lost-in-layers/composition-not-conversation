#!/usr/bin/env bash
# Download and evaluate the requested open VLMs sequentially on one GPU.
#
# Each checkpoint is downloaded into the repository's models/ directory,
# served on the same vLLM port,
# evaluated on all 14 Experiment-B conditions followed by Experiment A, and
# then unloaded before the next checkpoint starts. Existing successful runs
# are detected by the simulators and skipped, so the script is resumable.
# A combined Experiment A + B markdown report is written per model, right after
# that model finishes and before its weights are unloaded.
#
# Usage:
#   bash run_all_open_models.sh                       # all eight models
#   bash run_all_open_models.sh --skip-thinking-only # omit GLM-4.1V-Thinking
#
# Optional environment overrides:
#   GPU_ID=0 PORT=8201 MODELS_DIR=/path/to/models N_WORKERS=1 \
#   LOG_FOLDER=logs/Final-0 bash ...
#   TARGET_MAX_MODEL_LEN=65536 bash ...
#   MIN_MODEL_LEN=32768 bash ...   # override every entry's own context floor
#
# The judge cache is pinned inside LOG_FOLDER, so a fresh LOG_FOLDER is a fresh
# set of verdicts rather than a replay of an earlier run's cached ones.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SKIP_THINKING_ONLY=0
MODEL_FILTER=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-thinking-only) SKIP_THINKING_ONLY=1; shift ;;
    --only)
      # Run a subset without editing MODELS. The point is TP: entries needing
      # two GPUs want GPU_ID=0,1, and giving that to a TP=1 entry idles the
      # second card, so the two groups belong in separate launches.
      MODEL_FILTER="${2:-}"
      [[ -z "$MODEL_FILTER" ]] && { echo "--only needs a value" >&2; exit 2; }
      shift 2
      ;;
    -h|--help)
      sed -n '2,26p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash $0 [--skip-thinking-only] [--only alias[,alias...]]" >&2
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

export VLLM_PLUGINS=""
export PORT="${PORT:-8201}"
export VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"

GPU_ID="${GPU_ID:-0}"
N_WORKERS="${N_WORKERS:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TARGET_MAX_MODEL_LEN="${TARGET_MAX_MODEL_LEN:-65536}"
MIN_MODEL_LEN="${MIN_MODEL_LEN:-}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-1800}"
MODELS_DIR="${MODELS_DIR:-$SCRIPT_DIR/../../models}"
LOG_FOLDER="${LOG_FOLDER:-logs/Final-0}"
export LAYERED_VQA_JUDGE_CACHE="$SCRIPT_DIR/$LOG_FOLDER/judge_cache.jsonl"

D="../../data/derived/lic"
TEXT="$D/layered_vqa_text.json"
CANON="$D/layered_vqa_visual_canonical.json"
GROUND="$D/layered_vqa_grounding.json"
ORACLE="$D/layered_vqa_oracle_gold.json"
SERVER_LOG_DIR="$LOG_FOLDER/vllm_servers"
RUNTIME_DIR="$LOG_FOLDER/vllm_runtime"
mkdir -p "$MODELS_DIR" "$SERVER_LOG_DIR" "$RUNTIME_DIR" "$(dirname "$LAYERED_VQA_JUDGE_CACHE")"

for command_name in python hf vllm curl setsid; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command not found after activating $VENV_DIR: $command_name" >&2
    exit 1
  fi
done

for dataset in \
  "$TEXT" "$CANON" "$GROUND" "$ORACLE" \
  "$D/layered_vqa_aligned_vq.json" \
  "$D/layered_vqa_misaligned_vq.json" \
  "$D/layered_vqa_visual_evidence_first.json" \
  "$D/layered_vqa_visual_evidence_last.json"; do
  if [[ ! -f "$dataset" ]]; then
    echo "Required dataset not found: $dataset" >&2
    exit 1
  fi
done

# alias|HF repo|launch profile|context ceiling|context floor|thinking policy|TP
#
# context ceiling  upper bound this runner will not exceed even when the
#                  checkpoint supports more; the served length is
#                  min(ceiling, TARGET_MAX_MODEL_LEN, what the config declares).
#                  Serving far past what the conditions need only shrinks the
#                  KV cache.
# context floor    the shortest window this entry may run at. Below it the
#                  entry fails before serving instead of dropping its longest
#                  conversations one 400 at a time. It is per entry because the
#                  text towers differ: Qwen3-8B declares 40960, but
#                  InternLM2.5-7B and Qwen2.5-7B declare 32768, so a single
#                  global 40960 would reject the InternVL2.5/3 checkpoints for
#                  a window they can actually serve. SNOWBALL_V is the longest
#                  condition at ~37k tokens; 32768 entries are the ones to
#                  watch in the coverage column of their report.
# thinking policy  off           template takes enable_thinking; send false
#                  none          no thinking switch; send no template kwargs
#                  thinking-only dedicated reasoning checkpoint, cannot be off
# TP               tensor-parallel size. A property of the checkpoint, not of
#                  the launch: a 30B/32B checkpoint is ~60GB in bf16 and cannot
#                  fit one 64GB card once the KV cache is added, while forcing
#                  TP>1 on an 8B wastes a GPU and puts a collective on the
#                  critical path of every one of ~25k calls. GPU_ID must name at
#                  least TP devices ("GPU_ID=0,1"); the launch refuses rather
#                  than letting vLLM fail obscurely half a load in.
#
# The dedicated GLM-4.1V-Thinking checkpoint is intentionally last. Its
# official template exposes no supported non-thinking switch.
# Ordered by parameter count, smallest first, so a launch fails fast on a cheap
# checkpoint rather than after loading a 30B one, and so an interrupted run has
# covered the widest span of sizes. Commented entries are already in
# Final-0/1/2; uncomment an entry to include it in the next launch.
MODELS=(
  "qwen2.5-vl-3b|Qwen/Qwen2.5-VL-3B-Instruct|standard|65536|32768|none|1"
  "qwen3-vl-4b|Qwen/Qwen3-VL-4B-Instruct|standard|65536|40960|none|1"
  # "qwen2.5-vl-7b|Qwen/Qwen2.5-VL-7B-Instruct|standard|65536|32768|none|1"
  "qwen3-vl-8b|Qwen/Qwen3-VL-8B-Instruct|standard|65536|40960|none|1"
  # "internvl3.5-8b|OpenGVLab/InternVL3_5-8B|remote-code|65536|40960|off|1"
  # "ministral3-8b-instruct|mistralai/Ministral-3-8B-Instruct-2512-BF16|mistral|65536|40960|off|1"
  # "qwen3.5-9b|Qwen/Qwen3.5-9B|qwen|65536|40960|off|1"
  # "pixtral-12b|mistralai/Pixtral-12B-2409|mistral|65536|40960|none|1"
  # "qwen3.6-27b|Qwen/Qwen3.6-27B|qwen|65536|40960|off|1"
  # ~60-64GB of bf16 weights before any KV cache, so neither fits one 64GB
  # card; both are sharded over two. Run them in their own launch with
  # GPU_ID=0,1 -- the entries above are TP=1 and would each idle a second GPU.
  "qwen3-vl-30b-a3b|Qwen/Qwen3-VL-30B-A3B-Instruct|standard|65536|40960|none|2"
  "qwen2.5-vl-32b|Qwen/Qwen2.5-VL-32B-Instruct|standard|65536|32768|none|2"
)

VLLM_PID=""

stop_vllm() {
  if [[ -z "$VLLM_PID" ]]; then
    return
  fi

  if kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "Stopping vLLM process group $VLLM_PID ..."
    kill -TERM -- "-$VLLM_PID" 2>/dev/null || kill -TERM "$VLLM_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do
      if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "vLLM did not stop after 60 seconds; terminating its process group." >&2
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
  local alias="$1"
  local server_log="$2"
  local waited=0

  while (( waited < SERVER_WAIT_SECONDS )); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "vLLM exited while loading $alias. Last 100 log lines:" >&2
      tail -n 100 "$server_log" >&2 || true
      return 1
    fi

    if response="$(curl -fsS --max-time 5 "$VLLM_BASE_URL/models" 2>/dev/null)" && \
       python -c 'import json,sys; target=sys.argv[1]; data=json.load(sys.stdin); raise SystemExit(0 if any(x.get("id") == target for x in data.get("data", [])) else 1)' \
         "$alias" <<<"$response"; then
      echo "vLLM is ready: $alias"
      return 0
    fi

    sleep 2
    waited=$((waited + 2))
  done

  echo "Timed out after ${SERVER_WAIT_SECONDS}s waiting for $alias. Last 100 log lines:" >&2
  tail -n 100 "$server_log" >&2 || true
  return 1
}

resolve_max_model_len() {
  # Echo the context length to serve: the smallest of the requested target, the
  # model's per-entry ceiling and what the downloaded checkpoint declares.
  # Reading the config beats hardcoding a number per model — an entry that
  # cannot reach MIN_MODEL_LEN then fails loudly here instead of dropping the
  # longest conversations one 400 at a time.
  local model_path="$1"
  local ceiling="$2"
  local alias="$3"
  local floor="$4"
  local resolved

  # An explicit MIN_MODEL_LEN overrides every entry's own floor.
  if [[ -n "$MIN_MODEL_LEN" ]]; then
    floor="$MIN_MODEL_LEN"
  fi

  resolved="$(python - "$model_path" "$ceiling" "$TARGET_MAX_MODEL_LEN" <<'PY'
import json
import sys
from pathlib import Path

model_path, ceiling, target = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

# HF multimodal checkpoints put the text tower's window in different places:
# top level (Qwen, Gemma, Mistral), `text_config` (Gemma 3+/Molmo-style
# composite configs) or `llm_config` (InternVL). Mistral-format repos ship
# `params.json` instead. Take the largest declared window rather than guessing
# which key this family uses.
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
    if not path.exists():
        continue
    try:
        found.extend(windows(json.loads(path.read_text(encoding="utf-8"))))
    except (json.JSONDecodeError, OSError):
        continue

limits = [ceiling, target]
if found:
    limits.append(max(found))
else:
    # No declared window: serving the target and letting vLLM reject it beats
    # silently falling back to something short.
    print("could not read a context window from the checkpoint config; "
          "using the requested target", file=sys.stderr)
print(min(limits))
PY
)" || return 1

  if (( resolved < floor )); then
    echo "$alias resolves to a ${resolved}-token context, below its ${floor}-token floor." >&2
    echo "SNOWBALL_V prompts reach ~37k tokens, so this run would drop samples on a 400." >&2
    echo "Re-run with MIN_MODEL_LEN=$resolved if that is acceptable for this model." >&2
    return 1
  fi

  echo "$resolved"
}

resolve_served_name() {
  # The name vLLM must answer to is whatever models.yaml will put on the wire
  # for this alias, since that is what --models/--judge_model/--user_model send.
  # Reading it here also catches an alias missing from models.yaml now, rather
  # than after a 20-minute model load.
  local alias="$1"

  python - "$alias" <<'INNERPY'
import sys
from pathlib import Path

import yaml

alias = sys.argv[1]
config = yaml.safe_load(Path("models.yaml").read_text(encoding="utf-8"))
entry = (config.get("models") or {}).get(alias)
if entry is None:
    raise SystemExit(
        f"{alias} has no models.yaml entry; the simulators would fail to route "
        f"--models {alias} after the server was already up"
    )
print(entry.get("model", alias))
INNERPY
}

assert_chat_template() {
  # A checkpoint with no usable chat template does not fail at startup — vLLM
  # serves it and then rejects or mis-formats every /chat/completions call,
  # which reads downstream as a model that simply scored badly. Check first.
  local model_path="$1"
  local profile="$2"
  local alias="$3"

  python - "$model_path" "$profile" "$alias" <<'INNERPY'
import json
import sys
from pathlib import Path

model_path, profile, alias = Path(sys.argv[1]), sys.argv[2], sys.argv[3]

if profile == "mistral":
    # Mistral-format repos carry no Jinja template at all: --tokenizer_mode
    # mistral builds the prompt from the Mistral tokenizer definition, so that
    # file is what has to be present.
    found = [
        name
        for name in (
            "tekken.json",
            "tokenizer.model.v7m1",
            "tokenizer.model.v7",
            "tokenizer.model.v3",
            "tokenizer.model",
        )
        if (model_path / name).exists()
    ]
    if not found:
        raise SystemExit(
            f"{alias}: the mistral profile needs a Mistral tokenizer "
            f"(tekken.json or tokenizer.model*) in {model_path}; none found. "
            f"Either this repo is HF-format and wants the standard profile, or "
            f"the download is incomplete."
        )
    print(f"{alias}: mistral-format tokenizer ({found[0]})")
    raise SystemExit(0)

if profile == "phi-nothink":
    # make_phi_nothink_template rewrites this file and already fails loudly if
    # the generation prompt it patches has moved.
    if not (model_path / "chat_template.jinja").exists():
        raise SystemExit(f"{alias}: chat_template.jinja not found in {model_path}")
    print(f"{alias}: chat_template.jinja (patched for NOTHINK)")
    raise SystemExit(0)

# Everything else goes through vLLM's HF path, which resolves a template from
# these three places, newest convention first.
for name in ("chat_template.jinja", "chat_template.json"):
    if (model_path / name).exists():
        print(f"{alias}: {name}")
        break
else:
    config_path = model_path / "tokenizer_config.json"
    template = None
    if config_path.exists():
        try:
            template = json.loads(config_path.read_text(encoding="utf-8")).get(
                "chat_template"
            )
        except (json.JSONDecodeError, OSError):
            template = None
    if not template:
        raise SystemExit(
            f"{alias}: no chat template in {model_path} (looked for "
            f"chat_template.jinja, chat_template.json, and a chat_template key "
            f"in tokenizer_config.json). vLLM would serve this checkpoint and "
            f"then fail every chat request. Supply one with --chat-template, or "
            f"use a revision that ships it."
        )
    print(f"{alias}: chat_template in tokenizer_config.json")
INNERPY
}

make_phi_nothink_template() {
  local model_path="$1"
  local output="$RUNTIME_DIR/phi4_reasoning_vision_nothink.jinja"

  python - "$model_path/chat_template.jinja" "$output" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1])
target = Path(sys.argv[2])
text = source.read_text(encoding="utf-8")
old = "{% if add_generation_prompt %}{{ '<|im_start|>assistant<|im_sep|>' }}{% endif %}"
new = "{% if add_generation_prompt %}{{ '<|im_start|>assistant<|im_sep|><nothink>' }}{% endif %}"
if old not in text:
    raise SystemExit(f"Could not locate Phi generation prompt in {source}; refusing to run without forced NOTHINK")
target.write_text(text.replace(old, new, 1), encoding="utf-8")
print(target)
PY
}

start_vllm() {
  local alias="$1"
  local hf_id="$2"
  local profile="$3"
  local context_ceiling="$4"
  local context_floor="$5"
  local thinking_policy="$6"
  local model_path="$7"
  local tp="${8:-1}"
  local server_log="$SERVER_LOG_DIR/${alias}.log"
  local served_name
  local max_model_len
  local -a args

  # CUDA_VISIBLE_DEVICES is set from GPU_ID below, so GPU_ID is what bounds
  # the tensor-parallel size. Checking here turns "TP=2 on one GPU" into one
  # clear line instead of a vLLM stack trace after several minutes of loading.
  local -a visible_gpus
  IFS=',' read -ra visible_gpus <<< "$GPU_ID"
  if (( tp > ${#visible_gpus[@]} )); then
    echo "$alias needs tensor-parallel-size $tp but GPU_ID names only ${#visible_gpus[@]} device(s): $GPU_ID" >&2
    echo "Re-run that entry with e.g. GPU_ID=0,1" >&2
    return 1
  fi

  max_model_len="$(resolve_max_model_len "$model_path" "$context_ceiling" "$alias" "$context_floor")" || return 1
  echo "Context length for $alias: $max_model_len (target $TARGET_MAX_MODEL_LEN, ceiling $context_ceiling, floor $context_floor)"

  echo "Chat template: $(assert_chat_template "$model_path" "$profile" "$alias")" || return 1

  # Serve under whatever models.yaml routes this alias to. Some entries map to
  # the alias itself and some (qwen3.5-9b, qwen2.5-vl-7b) to the canonical HF
  # repo id; either way the served name has to match, or every request 404s on
  # an unknown model.
  served_name="$(resolve_served_name "$alias")" || return 1

  args=(
    "$model_path"
    --served-model-name "$served_name"
    --host 127.0.0.1
    --port "$PORT"
    --tensor-parallel-size "$tp"
    --max-model-len "$max_model_len"
    --max-num-seqs 1
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --generation-config vllm
    --limit-mm-per-prompt.image 24
  )

  case "$profile" in
    qwen)
      args+=(--reasoning-parser qwen3)
      ;;
    mistral)
      args+=(--tokenizer_mode mistral --config_format mistral --load_format mistral)
      ;;
    remote-code)
      args+=(--trust-remote-code)
      ;;
    phi-nothink)
      args+=(--trust-remote-code --dtype bfloat16 --chat-template "$(make_phi_nothink_template "$model_path")")
      ;;
    internvl)
      args+=(--trust-remote-code --chat-template-content-format string)
      ;;
    standard) ;;
    *)
      echo "Unknown vLLM launch profile: $profile" >&2
      return 1
      ;;
  esac

  case "$thinking_policy" in
    off)
      # Only for templates that actually branch on it. phi-nothink forces the
      # same thing through its patched template instead.
      if [[ "$profile" != "phi-nothink" ]]; then
        args+=(--default-chat-template-kwargs '{"enable_thinking": false}')
      fi
      ;;
    none)
      # No thinking switch in this template. Passing enable_thinking anyway
      # would either be silently dropped or, on a strict template, raise on
      # every request — so send nothing. These checkpoints answer directly.
      ;;
    thinking-only)
      echo "WARNING: $hf_id is a dedicated thinking checkpoint; thinking cannot be disabled." >&2
      ;;
    *)
      echo "Unknown thinking policy: $thinking_policy" >&2
      return 1
      ;;
  esac

  if curl -fsS --max-time 3 "$VLLM_BASE_URL/models" >/dev/null 2>&1; then
    echo "Port $PORT already has an OpenAI-compatible server. Stop it before running this script." >&2
    return 1
  fi

  echo "Starting $hf_id as $alias; server log: $server_log"
  setsid env HIP_VISIBLE_DEVICES="$GPU_ID" CUDA_VISIBLE_DEVICES="$GPU_ID" \
    vllm serve "${args[@]}" >"$server_log" 2>&1 &
  VLLM_PID=$!
  wait_for_server "$served_name" "$server_log"
}

run_exp_b() {
  local model="$1"
  local -a M=(--models "$model" --judge_model "$model" --user_model "$model")
  local F

  echo "[$model] Experiment B: 1/14 FULL"
  LAYERED_VQA_DATASET="$TEXT" python run_simulations.py --dataset_file "$TEXT" --tasks layered_vqa "${M[@]}" --N_full_runs 1 --N_concat_runs 0 --N_sharded_runs 0 --N_shuffle_concat_runs 0 --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"

  echo "[$model] Experiment B: 2/14 CONCAT_Q"
  LAYERED_VQA_DATASET="$TEXT" python run_simulations.py --dataset_file "$TEXT" --tasks layered_vqa "${M[@]}" --N_full_runs 0 --N_concat_runs 1 --N_sharded_runs 0 --N_shuffle_concat_runs 0 --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"

  echo "[$model] Experiment B: 3/14 SHARDED_Q"
  LAYERED_VQA_DATASET="$TEXT" python run_simulations.py --dataset_file "$TEXT" --tasks layered_vqa "${M[@]}" --user_agent llm --N_full_runs 0 --N_concat_runs 0 --N_sharded_runs 1 --N_shuffle_concat_runs 0 --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"

  echo "[$model] Experiment B: 4/14 RECAP_Q"
  LAYERED_VQA_DATASET="$TEXT" python simulator_recap.py --task layered_vqa --models "$model" --judge_model "$model" --sharded_conv_types sharded --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"

  echo "[$model] Experiment B: 5/14 SNOWBALL_Q"
  LAYERED_VQA_DATASET="$TEXT" python simulator_snowball.py --task layered_vqa "${M[@]}" --N_snowball_runs 1 --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"

  echo "[$model] Experiment B: 6/14 CONCAT_V"
  LAYERED_VQA_DATASET="$CANON" python run_simulations.py --dataset_file "$CANON" --tasks layered_vqa "${M[@]}" --N_full_runs 1 --N_concat_runs 0 --N_sharded_runs 0 --N_shuffle_concat_runs 0 --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"

  echo "[$model] Experiment B: 7/14 SHARDED_V"
  LAYERED_VQA_DATASET="$CANON" python run_simulations.py --dataset_file "$CANON" --tasks layered_vqa "${M[@]}" --user_agent scripted --N_full_runs 0 --N_concat_runs 0 --N_sharded_runs 1 --N_shuffle_concat_runs 0 --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"

  echo "[$model] Experiment B: 8/14 RECAP_V"
  LAYERED_VQA_DATASET="$CANON" python simulator_recap.py --task layered_vqa --models "$model" --judge_model "$model" --sharded_conv_types sharded-scripted --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"

  echo "[$model] Experiment B: 9/14 SNOWBALL_V"
  LAYERED_VQA_DATASET="$CANON" python simulator_snowball.py --task layered_vqa --user_agent scripted "${M[@]}" --N_snowball_runs 1 --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"

  echo "[$model] Experiment B: 10/14 SHARDED_VQ aligned"
  F="$D/layered_vqa_aligned_vq.json"
  LAYERED_VQA_DATASET="$F" python run_simulations.py --dataset_file "$F" --tasks layered_vqa "${M[@]}" --user_agent scripted --N_full_runs 0 --N_concat_runs 0 --N_sharded_runs 1 --N_shuffle_concat_runs 0 --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"

  echo "[$model] Experiment B: 11/14 SHARDED_VQ misaligned"
  F="$D/layered_vqa_misaligned_vq.json"
  LAYERED_VQA_DATASET="$F" python run_simulations.py --dataset_file "$F" --tasks layered_vqa "${M[@]}" --user_agent scripted --N_full_runs 0 --N_concat_runs 0 --N_sharded_runs 1 --N_shuffle_concat_runs 0 --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"

  echo "[$model] Experiment B: 12/14 EVIDENCE_FIRST_V"
  F="$D/layered_vqa_visual_evidence_first.json"
  LAYERED_VQA_DATASET="$F" python run_simulations.py --dataset_file "$F" --tasks layered_vqa "${M[@]}" --user_agent scripted --N_full_runs 0 --N_concat_runs 0 --N_sharded_runs 1 --N_shuffle_concat_runs 0 --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"

  echo "[$model] Experiment B: 13/14 EVIDENCE_LAST_V"
  F="$D/layered_vqa_visual_evidence_last.json"
  LAYERED_VQA_DATASET="$F" python run_simulations.py --dataset_file "$F" --tasks layered_vqa "${M[@]}" --user_agent scripted --N_full_runs 0 --N_concat_runs 0 --N_sharded_runs 1 --N_shuffle_concat_runs 0 --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"

  echo "[$model] Experiment B: 14/14 SHUFFLE_Q-CONCAT"
  LAYERED_VQA_DATASET="$TEXT" python run_simulations.py --dataset_file "$TEXT" --tasks layered_vqa "${M[@]}" --N_full_runs 0 --N_concat_runs 0 --N_sharded_runs 0 --N_shuffle_concat_runs 1 --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"
}

run_exp_a() {
  local model="$1"
  local -a M=(--models "$model" --judge_model "$model" --user_model "$model")
  local V F

  echo "[$model] Experiment A: A1, A2 and A3"
  # A1 is identical to Experiment-B FULL and therefore resumes/skips here.
  for V in text grounding oracle_gold; do
    F="$D/layered_vqa_${V}.json"
    LAYERED_VQA_DATASET="$F" python run_simulations.py --dataset_file "$F" --tasks layered_vqa "${M[@]}" --N_full_runs 1 --N_concat_runs 0 --N_sharded_runs 0 --N_shuffle_concat_runs 0 --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"
  done
}

write_model_report() {
  # One markdown file per model, holding both experiments, written while that
  # model is still the current one. A multi-day multi-model run then leaves a
  # readable result file behind at every step rather than only at the end.
  local model="$1"
  local report="$LOG_FOLDER/results_${model}.md"

  python score_model_report.py --log_folder "$LOG_FOLDER" --models "$model" \
    --out "$report"
  echo "[$model] Experiment A + B summary: $report"
}

echo "Model directory: $(cd "$MODELS_DIR" && pwd)"
echo "GPU: $GPU_ID"
echo "vLLM endpoint: $VLLM_BASE_URL"
echo "vLLM version: $(vllm --version 2>&1 | head -n 1)"

for spec in "${MODELS[@]}"; do
  IFS='|' read -r alias hf_id profile context_ceiling context_floor thinking_policy tp <<<"$spec"
  tp="${tp:-1}"   # entries written before the TP field mean one GPU

  if [[ -n "$MODEL_FILTER" && ",$MODEL_FILTER," != *",$alias,"* ]]; then
    continue
  fi

  if [[ "$thinking_policy" == "thinking-only" && "$SKIP_THINKING_ONLY" -eq 1 ]]; then
    echo "Skipping thinking-only model: $hf_id"
    continue
  fi

  echo
  echo "============================================================================="
  echo "MODEL: $hf_id"
  echo "ALIAS: $alias"
  echo "THINKING: $thinking_policy"
  echo "============================================================================="

  model_path="$MODELS_DIR/$alias"
  echo "Downloading/resuming $hf_id -> $model_path ..."
  hf download "$hf_id" --local-dir "$model_path"
  echo "Local model: $model_path"

  start_vllm "$alias" "$hf_id" "$profile" "$context_ceiling" "$context_floor" "$thinking_policy" "$model_path" "$tp"
  python agent_config.py --model "$alias"

  run_exp_b "$alias"
  run_exp_a "$alias"
  write_model_report "$alias"

  stop_vllm
done

echo
echo "All requested model runs are complete."
