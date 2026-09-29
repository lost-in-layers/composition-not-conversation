#!/usr/bin/env bash
# The 14 Experiment-B delivery conditions and the 3 Experiment-A conditions, in
# one place, so a runner and a smoke test cannot drift apart on what a
# condition actually is.
#
# Sourced, never executed. The caller sets, before sourcing:
#
#   D           dataset directory (holds the eight layered_vqa_*.json files)
#   LOG_FOLDER  where conversations and the judge cache land
#   N_WORKERS   simulator concurrency
#
# and may set:
#
#   DRY_RUN=1      print each command instead of running it
#   KEEP_GOING=1   record a failing condition and continue, instead of aborting
#   SKIP_POSTHOC=1 skip the behavioural pass run_posthoc adds after a model
#
# ../../docs/lic_simulation.md is the document these commands implement; keep
# the two in step.

D="${D:-../../data/derived/lic}"
TEXT="$D/layered_vqa_text.json"
CANON="$D/layered_vqa_visual_canonical.json"
GROUND="$D/layered_vqa_grounding.json"
ORACLE="$D/layered_vqa_oracle_gold.json"

DRY_RUN="${DRY_RUN:-0}"
KEEP_GOING="${KEEP_GOING:-0}"

# Appended to by run_step, model included because a multi-model summary of
# bare condition names is 17 lines repeated N times:
#   "PASS [gpt-5.6-luna] Experiment B: 3/14 SHARDED_Q"
CONDITION_RESULTS=()

assert_datasets() {
  local dataset
  for dataset in \
    "$TEXT" "$CANON" "$GROUND" "$ORACLE" \
    "$D/layered_vqa_aligned_vq.json" \
    "$D/layered_vqa_misaligned_vq.json" \
    "$D/layered_vqa_visual_evidence_first.json" \
    "$D/layered_vqa_visual_evidence_last.json"; do
    if [[ ! -f "$dataset" ]]; then
      echo "Required dataset not found: $dataset" >&2
      return 1
    fi
  done
}

run_step() {
  # One condition. The command is passed as argv rather than as a string, so a
  # dry run prints exactly what would execute, quoting included.
  local model="$1" label="$2"
  shift 2

  echo "[$model] $label"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '    '
    printf '%q ' "$@"
    printf '\n'
    CONDITION_RESULTS+=("DRY  [$model] $label")
    return 0
  fi

  local status=0
  "$@" || status=$?
  if (( status == 0 )); then
    CONDITION_RESULTS+=("PASS [$model] $label")
    return 0
  fi

  CONDITION_RESULTS+=("FAIL [$model] $label (exit $status)")
  echo "[$model] $label FAILED with exit $status" >&2
  if [[ "$KEEP_GOING" == "1" ]]; then
    return 0
  fi
  return "$status"
}

run_exp_b() {
  # Both halves of a condition's identity travel together: LAYERED_VQA_DATASET
  # is what the task, simulator_recap and simulator_snowball see, and
  # --dataset_file is what run_simulations.py loads. A mismatch runs fine and
  # answers the wrong condition.
  local model="$1"
  local -a M=(--models "$model" --judge_model "$model" --user_model "$model")
  local F

  run_step "$model" "Experiment B: 1/14 FULL" \
    env LAYERED_VQA_DATASET="$TEXT" python run_simulations.py --dataset_file "$TEXT" --tasks layered_vqa "${M[@]}" --N_full_runs 1 --N_concat_runs 0 --N_sharded_runs 0 --N_shuffle_concat_runs 0 --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"

  run_step "$model" "Experiment B: 2/14 CONCAT_Q" \
    env LAYERED_VQA_DATASET="$TEXT" python run_simulations.py --dataset_file "$TEXT" --tasks layered_vqa "${M[@]}" --N_full_runs 0 --N_concat_runs 1 --N_sharded_runs 0 --N_shuffle_concat_runs 0 --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"

  run_step "$model" "Experiment B: 3/14 SHARDED_Q" \
    env LAYERED_VQA_DATASET="$TEXT" python run_simulations.py --dataset_file "$TEXT" --tasks layered_vqa "${M[@]}" --user_agent llm --N_full_runs 0 --N_concat_runs 0 --N_sharded_runs 1 --N_shuffle_concat_runs 0 --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"

  # RECAP replays finished sharded logs, so it has to follow 3/14 and 7/14.
  run_step "$model" "Experiment B: 4/14 RECAP_Q" \
    env LAYERED_VQA_DATASET="$TEXT" python simulator_recap.py --task layered_vqa --models "$model" --judge_model "$model" --sharded_conv_types sharded --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"

  run_step "$model" "Experiment B: 5/14 SNOWBALL_Q" \
    env LAYERED_VQA_DATASET="$TEXT" python simulator_snowball.py --task layered_vqa "${M[@]}" --N_snowball_runs 1 --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"

  run_step "$model" "Experiment B: 6/14 CONCAT_V" \
    env LAYERED_VQA_DATASET="$CANON" python run_simulations.py --dataset_file "$CANON" --tasks layered_vqa "${M[@]}" --N_full_runs 1 --N_concat_runs 0 --N_sharded_runs 0 --N_shuffle_concat_runs 0 --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"

  run_step "$model" "Experiment B: 7/14 SHARDED_V" \
    env LAYERED_VQA_DATASET="$CANON" python run_simulations.py --dataset_file "$CANON" --tasks layered_vqa "${M[@]}" --user_agent scripted --N_full_runs 0 --N_concat_runs 0 --N_sharded_runs 1 --N_shuffle_concat_runs 0 --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"

  run_step "$model" "Experiment B: 8/14 RECAP_V" \
    env LAYERED_VQA_DATASET="$CANON" python simulator_recap.py --task layered_vqa --models "$model" --judge_model "$model" --sharded_conv_types sharded-scripted --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"

  run_step "$model" "Experiment B: 9/14 SNOWBALL_V" \
    env LAYERED_VQA_DATASET="$CANON" python simulator_snowball.py --task layered_vqa --user_agent scripted "${M[@]}" --N_snowball_runs 1 --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"

  F="$D/layered_vqa_aligned_vq.json"
  run_step "$model" "Experiment B: 10/14 SHARDED_VQ aligned" \
    env LAYERED_VQA_DATASET="$F" python run_simulations.py --dataset_file "$F" --tasks layered_vqa "${M[@]}" --user_agent scripted --N_full_runs 0 --N_concat_runs 0 --N_sharded_runs 1 --N_shuffle_concat_runs 0 --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"

  F="$D/layered_vqa_misaligned_vq.json"
  run_step "$model" "Experiment B: 11/14 SHARDED_VQ misaligned" \
    env LAYERED_VQA_DATASET="$F" python run_simulations.py --dataset_file "$F" --tasks layered_vqa "${M[@]}" --user_agent scripted --N_full_runs 0 --N_concat_runs 0 --N_sharded_runs 1 --N_shuffle_concat_runs 0 --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"

  F="$D/layered_vqa_visual_evidence_first.json"
  run_step "$model" "Experiment B: 12/14 EVIDENCE_FIRST_V" \
    env LAYERED_VQA_DATASET="$F" python run_simulations.py --dataset_file "$F" --tasks layered_vqa "${M[@]}" --user_agent scripted --N_full_runs 0 --N_concat_runs 0 --N_sharded_runs 1 --N_shuffle_concat_runs 0 --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"

  F="$D/layered_vqa_visual_evidence_last.json"
  run_step "$model" "Experiment B: 13/14 EVIDENCE_LAST_V" \
    env LAYERED_VQA_DATASET="$F" python run_simulations.py --dataset_file "$F" --tasks layered_vqa "${M[@]}" --user_agent scripted --N_full_runs 0 --N_concat_runs 0 --N_sharded_runs 1 --N_shuffle_concat_runs 0 --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"

  run_step "$model" "Experiment B: 14/14 SHUFFLE_Q-CONCAT" \
    env LAYERED_VQA_DATASET="$TEXT" python run_simulations.py --dataset_file "$TEXT" --tasks layered_vqa "${M[@]}" --N_full_runs 0 --N_concat_runs 0 --N_sharded_runs 0 --N_shuffle_concat_runs 1 --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"
}

run_exp_a() {
  # A1 is the same run as Experiment-B FULL and therefore resumes/skips here;
  # A2 and A3 differ from it only by dataset_fn, which get_run_counts filters
  # on, so the three resume independently without a new conv_type.
  local model="$1"
  local -a M=(--models "$model" --judge_model "$model" --user_model "$model")
  local variant F index=0

  for variant in text grounding oracle_gold; do
    index=$((index + 1))
    F="$D/layered_vqa_${variant}.json"
    run_step "$model" "Experiment A: A${index} ${variant}" \
      env LAYERED_VQA_DATASET="$F" python run_simulations.py --dataset_file "$F" --tasks layered_vqa "${M[@]}" --N_full_runs 1 --N_concat_runs 0 --N_sharded_runs 0 --N_shuffle_concat_runs 0 --N_workers "$N_WORKERS" --log_folder "$LOG_FOLDER"
  done
}

write_model_report() {
  # One markdown file per model holding both experiments, written as soon as
  # that model finishes rather than only at the end of a multi-model run.
  local model="$1"
  local report="$LOG_FOLDER/results_${model}.md"

  run_step "$model" "Experiment A + B report" \
    python score_model_report.py --log_folder "$LOG_FOLDER" --models "$model" \
      --dataset_dir "$D" --out "$report"
  echo "[$model] Experiment A + B summary: $report"
}

run_posthoc() {
  # The behavioural measures, for this model's own logs only.
  #
  # The simulators write one bit per conversation: is_correct over the whole
  # transcript. That answers "did the reference answer appear anywhere" and
  # nothing else -- not which turn carried it, not whether the evidence
  # supporting it had been delivered yet, not how many times the model changed
  # its mind. posthoc_judge.py asks for all of that in one further judge call
  # per conversation, and posthoc_analysis.py turns the result into the
  # behavioural tables: first-attempt position, verbosity quintiles, answer
  # bloat, the layer-citation matrix, grounded accuracy (ungrounded_correct)
  # and answer churn.
  #
  # --assistants pins it to this model's own conversations and --judges to
  # itself as grader, so this is the diagonal only. The judge x assistant
  # matrix is a separate, much larger pass (run_posthoc_judging.sh) and is
  # deliberately not run here -- filling it for N models costs N^2 passes, and
  # for metered models that is a bill, not an afternoon. The cross-judge table
  # posthoc_analysis.py prints will therefore be 1x1; every other table is
  # complete.
  #
  # It is resumable in its own right: outputs are append-only and keyed by
  # conv_id, so an interrupted pass skips what it already graded. Errored
  # gradings land in primary-analysis/<judge>/errors.jsonl and are retried next
  # time rather than frozen into the results as false negatives.
  #
  # Filed under primary-analysis/, not posthoc/: these numbers come out of the
  # primary run, and a folder named "posthoc" beside them would claim they were
  # produced later by a separate pass. logs/*/posthoc/ stays reserved for the
  # cross-judge matrix run_posthoc_judging.sh fills in, so the two never mix in
  # one folder.
  local model="$1"
  local report="$LOG_FOLDER/primary_analysis_${model}.md"
  local csv_dir="$LOG_FOLDER/primary_analysis_csv_${model}"

  if [[ "${SKIP_POSTHOC:-0}" == "1" ]]; then
    echo "[$model] SKIP_POSTHOC=1: behavioural pass skipped"
    return 0
  fi

  # --data_dir is $D, not the default: on a smoke run the gold_layer_ids and
  # layer labels these measures need have to come from the same cut-down
  # datasets the conversations were produced from.
  run_step "$model" "Analysis: re-grade own logs (attempts, turns, grounding)" \
    python posthoc_judge.py --log_folder "$LOG_FOLDER" --judge_model "$model" \
      --assistants "$model" --data_dir "$D" --subdir primary-analysis \
      --workers "$N_WORKERS"

  run_step "$model" "Analysis: behavioural tables" \
    python posthoc_analysis.py --runs "$LOG_FOLDER" --judges "$model" \
      --data_dir "$D" --subdir primary-analysis --out "$report" \
      --csv_dir "$csv_dir"

  echo "[$model] Behavioural report: $report"
  echo "[$model] Behavioural CSVs:   $csv_dir/"
}

print_condition_summary() {
  local line
  echo
  echo "-----------------------------------------------------------------------------"
  echo "CONDITION SUMMARY"
  echo "-----------------------------------------------------------------------------"
  for line in "${CONDITION_RESULTS[@]}"; do
    echo "  $line"
  done
  echo "  total: ${#CONDITION_RESULTS[@]}"
}

failed_condition_count() {
  local line count=0
  for line in "${CONDITION_RESULTS[@]}"; do
    [[ "$line" == FAIL* ]] && count=$((count + 1))
  done
  echo "$count"
}
