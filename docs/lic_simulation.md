# Running Layered-VQA through the conversation simulator

This wires the human-reviewed question shards into the multi-turn simulator so
the conditions in [README-EXP.md](../README-EXP.md) can be run.

## 1. Export the datasets

```sh
python src/vecgen/export_lic_dataset.py --all
```

Writes eight sample files plus a `.manifest.json` each to `data/derived/lic/`,
and flattens every RGBA layer into `data/derived/lic_renders/`.

| File | Serves |
|---|---|
| `layered_vqa_text.json` | FULL (Exp-A condition 1), CONCAT_Q, SHARDED_Q, RECAP_Q, SNOWBALL_Q, SHUFFLE_Q-CONCAT |
| `layered_vqa_grounding.json` | Exp-A condition 2 (full img + all layers + answer + layer selection) |
| `layered_vqa_oracle_gold.json` | Exp-A condition 3 (oracle gold layers) |
| `layered_vqa_aligned_vq.json` | SHARDED_VQ (aligned) |
| `layered_vqa_misaligned_vq.json` | SHARDED_VQ (misaligned) |
| `layered_vqa_visual_canonical.json` | CONCAT_V, SHARDED_V, SNOWBALL_V, RECAP_V |
| `layered_vqa_visual_evidence_first.json` | EVIDENCE_FIRST_V |
| `layered_vqa_visual_evidence_last.json` | EVIDENCE_LAST_V |

Useful flags: `--limit 5` for a pilot slice, `--categories spatial counting`,
`--opener-style raw` to keep annotator-voice intent shards, `--flatten-mode
background` to composite layers over the background instead of white,
`--skip-render` once the renders exist.

### Why layers are re-rendered

Instance layers are stored RGBA with real transparency. Sent as-is, a vision
API composites alpha onto black and the model receives a near-black frame:
the source `layer_1.png` of `lvis_000000108442` has mean luma **2** when its
alpha is dropped, and **253** after flattening. Never point the simulator at
the raw layer PNGs.

## 2. Run simulations

All commands run from `vendor/conv-smi/` — the code opens its
prompts by relative path. `LAYERED_VQA_DATASET` selects the exported variant
and must be set, because `simulator_recap` reloads samples through the task.

```sh
cd vendor/conv-smi
export VLLM_BASE_URL=http://localhost:8000/v1
export OPENROUTER_API_KEY=...          # only needed for closed-weight models
export LAYERED_VQA_DATASET=../../data/derived/lic/layered_vqa_text.json
```

### Choosing models

Every model goes through the OpenAI chat-completions API: open weights via the
local vLLM server, closed weights via OpenRouter. `vendor/conv-smi/models.yaml`
maps each name you pass to `--models` / `--judge_model` / `--user_model` onto a
backend and the id sent on the wire:

```yaml
local-vlm:          {backend: vllm,       model: local-vlm}
claude-sonnet-4.5:  {backend: openrouter, model: anthropic/claude-sonnet-4.5}
```

Add a model by editing that file, or pass `backend:id` inline for a one-off
(`--judge_model openrouter:openai/gpt-4o-mini`). An unrecognised name is a hard
error rather than a guess, and every run prints one line per model saying where
its calls go:

```
[models] local-vlm -> vllm:local-vlm
[models] claude-sonnet-4.5 -> openrouter:anthropic/claude-sonnet-4.5
```

Nothing tracks spend — check OpenRouter's dashboard for that.

The commands below use `local-vlm` for every role. Swapping any of them to a
closed model is a matter of naming it — running the assistant on OpenRouter
while the judge stays local is a normal setup, and a sensible one, since the
judge is the cheaper role to keep free.

Serve a vision model with enough image slots for the widest *conversation*,
not just the widest turn: SNOWBALL_V re-attaches every revealed layer each
turn, so a 6-layer scene peaks at 0+1+…+6 = 21 images in one prompt.

```sh
vllm serve <vision-model> --served-model-name local-vlm --limit-mm-per-prompt.image 24
```

### Experiment A — layer-grounded VQA

All three conditions are single-turn, so all three are `full`-type runs; they are
told apart by the dataset they ran on, exactly as CONCAT_V is. Condition 1 is the
FULL run below and needs nothing extra.

```sh
# A2 — full image + background + every retained layer, answer AND layer ids
LAYERED_VQA_DATASET=../../data/derived/lic/layered_vqa_grounding.json \
python run_simulations.py \
  --dataset_file ../../data/derived/lic/layered_vqa_grounding.json \
  --tasks layered_vqa --models local-vlm --judge_model local-vlm \
  --N_full_runs 1 --N_concat_runs 0 --N_sharded_runs 0 --N_shuffle_concat_runs 0

# A3 — oracle: only a gold minimal sufficient layer set, answer only
LAYERED_VQA_DATASET=../../data/derived/lic/layered_vqa_oracle_gold.json \
python run_simulations.py \
  --dataset_file ../../data/derived/lic/layered_vqa_oracle_gold.json \
  --tasks layered_vqa --models local-vlm --judge_model local-vlm \
  --N_full_runs 1 --N_concat_runs 0 --N_sharded_runs 0 --N_shuffle_concat_runs 0
```

Only the `grounding` samples carry `meta.response_format: answer_and_layers`, which
makes the task serve the layer-selection prompt — the model must answer *and* list
the layer ids it used, in `Answer:` / `Layers:` / `Confidence:` / `Why:` lines. Its
predicted set is scored deterministically against `gold_layer_ids`,
`minimal_sufficient_layer_sets` and `distractor_layer_ids`, and the metrics are
written into each conversation's `answer-evaluation` trace entry next to the answer
verdict. A3 uses the ordinary answer-only prompt and is graded as usual.

Aggregate the three conditions into README-EXP §1's metrics table:

```sh
python score_experiment_a.py                 # markdown to stdout
python score_experiment_a.py --csv exp_a.csv # one row per conversation
```

The CSV is the input for the paired tests and bootstrap CIs of README-EXP §5; the
script itself reports point estimates only.

`score_experiment_b.py` does the same for the 14 delivery conditions. Each one
is identified by a (conv_type, dataset) pair — `sharded-scripted` alone holds
SHARDED_V, both SHARDED_VQ arms and both EVIDENCE_* arms — and every accuracy
comes with a coverage figure, because a conversation that overflowed the
server's context window is missing from the logs rather than scored wrong.
`score_model_report.py --models <alias>` renders both experiments into one
markdown file per model; `run_all_open_models.sh` calls it as each model
finishes.

### FULL, CONCAT_Q, SHARDED_Q and SHUFFLE_Q-CONCAT

```sh
python run_simulations.py \
  --dataset_file ../../data/derived/lic/layered_vqa_text.json \
  --tasks layered_vqa --models local-vlm \
  --judge_model local-vlm --user_model local-vlm \
  --N_full_runs 1 --N_concat_runs 1 --N_sharded_runs 1 \
  --N_shuffle_concat_runs 1 --N_workers 4
```

`--N_shuffle_concat_runs` is SHUFFLE_Q-CONCAT: one turn, intent shard first,
the remaining shards in a fresh random order per run (the realized order is in
the logged user turn). The sample dict itself is never mutated.

SHUFFLED_SHARDED_Q — the same dataset with the shards *revealed* in a random
order — was dropped from Experiment B: SHUFFLE_Q-CONCAT already varies shard
order, and doing it again across turns confounded order with the LLM user
agent's reveal choices. The `scripted-shuffle` user agent that served it is
still available (`--user_agent scripted-shuffle`, logging under
`sharded-scripted-shuffle`) for anyone who wants to run it ad hoc.

### CONCAT_V

The full question and *all* layers in one turn is a `full`-type run on the
canonical visual export (its `question` field is the whole question and its
shards carry every layer):

```sh
LAYERED_VQA_DATASET=../../data/derived/lic/layered_vqa_visual_canonical.json \
python run_simulations.py \
  --dataset_file ../../data/derived/lic/layered_vqa_visual_canonical.json \
  --tasks layered_vqa --models local-vlm \
  --judge_model local-vlm --user_model local-vlm \
  --N_full_runs 1 --N_concat_runs 0 --N_sharded_runs 0
```

Do **not** use `aligned_vq` for this: that file carries only the supporting
(gold) layers, so a concat run on it is a different condition — an
evidence-only control, worth running, but not CONCAT_V.

Nor is `visual_canonical` a substitute for the Experiment A grounding file: it
carries **no** full RGB, keeps its layers on the shards rather than at sample
level, and asks for the answer alone. CONCAT_V is "full question + all layers";
Exp-A condition 2 is "full image + all layers + answer + layer selection".

### SHARDED_V

`--N_sharded_runs 1` on `layered_vqa_visual_canonical.json` **with
`--user_agent scripted`**. The scripted agent delivers the canonical arrival
order that defines this condition: the layer shards all read "here's another
part of the image", so an LLM user agent picks among near-indistinguishable
shards and improvises its own order.

On any dataset with per-shard images the drivers refuse to *default* the
agent — `--user_agent` must be passed explicitly. An explicit `--user_agent
llm` is accepted as its own condition (improvised reveal order, announced
with a `[warn]` line): the agent sees each shard's attachments by neutral
label (`"attaches": ["layer 2"]`, never a filename) and the shard it picks
brings its image along. Those runs log under `sharded` (the llm conv_type),
so they never mix with `sharded-scripted` logs.

### Aligned vs. misaligned SHARDED_VQ

Reveal order is the manipulation here, so `--user_agent scripted` replaces the
improvising user agent with one that walks the exported schedule.

```sh
for V in aligned_vq misaligned_vq; do
  LAYERED_VQA_DATASET=../../data/derived/lic/layered_vqa_$V.json \
  python run_simulations.py \
    --dataset_file ../../data/derived/lic/layered_vqa_$V.json \
    --tasks layered_vqa --models local-vlm \
    --judge_model local-vlm --user_model local-vlm \
    --user_agent scripted \
    --N_full_runs 0 --N_concat_runs 0 --N_sharded_runs 1
done
```

Misalignment is enforced against `question_shard_support`: every layer group is reassigned to a shard none of its layers support. 
### EVIDENCE_FIRST_V and EVIDENCE_LAST_V

Same shape, pointed at `layered_vqa_visual_evidence_{first,last}.json`, also
with `--user_agent scripted`.

### SNOWBALL_Q and SNOWBALL_V

`run_simulations.py` only drives FULL, CONCAT and SHARDED. Snowball has its own
entry point, which reads samples through the task rather than `--dataset_file`,
so `LAYERED_VQA_DATASET` is the only thing selecting the variant.

```sh
# SNOWBALL_Q
LAYERED_VQA_DATASET=../../data/derived/lic/layered_vqa_text.json \
python simulator_snowball.py --task layered_vqa \
  --models local-vlm --judge_model local-vlm --user_model local-vlm \
  --N_snowball_runs 1 --N_workers 4

# SNOWBALL_V — reveal order is the manipulation, so script the user agent
LAYERED_VQA_DATASET=../../data/derived/lic/layered_vqa_visual_canonical.json \
python simulator_snowball.py --task layered_vqa --user_agent scripted \
  --models local-vlm --judge_model local-vlm --user_model local-vlm \
  --N_snowball_runs 1 --N_workers 4
```

A snowball turn repeats everything so far: the accumulated hint texts *and*
every already-revealed image, re-attached alongside the new layer (turn k of a
visual run carries k−1 repeated layers plus the new one). Sample-level images
(the full RGB in the text variant) still arrive once, on the first turn.

Scripted runs log as `snowball-scripted`, so they are counted separately from
improvised `snowball` runs and the two never resume each other's work.

### RECAP_Q and RECAP_V

Recap replays finished SHARDED logs: run the matching SHARDED condition first
and point `LAYERED_VQA_DATASET` at the same file it used. The driver picks up
sharded conversations whose logged `dataset_fn` matches that file's basename,
scanning both `sharded` and `sharded-scripted` folders (override with
`--sharded_conv_types`), and skips conversations it has already recapped.

```sh
# RECAP_Q — after SHARDED_Q
LAYERED_VQA_DATASET=../../data/derived/lic/layered_vqa_text.json \
python simulator_recap.py --task layered_vqa --models local-vlm

# RECAP_V — after SHARDED_V
LAYERED_VQA_DATASET=../../data/derived/lic/layered_vqa_visual_canonical.json \
python simulator_recap.py --task layered_vqa --models local-vlm
```

The recap turn re-sends the **full RGB image** (the exporter writes it as
`recap_images` on every sample) with a prompt naming exactly that attachment.
Text-carrying variants restate their shards; visual variants restate the full
question, since their shards are filler captions. `recap-full` exists in the
code but no CLI path reaches it.

## 3. Inspect

```sh
streamlit run app_conv_viewer.py
```

## Design notes

**Information matching.** Aligned and misaligned deliver the identical layer
set in the identical per-turn counts; only which shard each layer accompanies
differs, so the contrast isolates alignment rather than confounding it with
how much evidence arrived. The three visual orderings likewise deliver the
same layer set, differing only in arrival order.

**Layer naming.** Prompts identify layers by number only. The reviewed
`label` is frequently the answer itself ("computer mouse"), so it would let a
model answer from text alone; labels live in `meta.layer_labels` for analysis.

**Opener voice.** `user_agent.py` sends the first shard verbatim, with no LLM
rephrasing. The raw intent shards are annotator-voice ("Determine a property
of a particular object in the image."), so the exporter rewrites them into
user voice by default. `--opener-style raw` disables this. A one-time LLM
rewrite pass with human spot-checks would be better than the current rule.

**Turns with no new image.** About a quarter of shard-to-layer support entries
map to no layer, so under aligned delivery some turns add text and no image.
That is faithful to the annotation, not an export bug.

**Grading.** Answers are free-form, so `evaluate_conversation` runs an LLM
equivalence judge; every verdict is the LLM's (no exact-match shortcut — the
only deterministic outcome is an empty transcript, scored incorrect).
A conversation is graded **once, when it ends**, from its whole transcript —
there is no turn classifier and no answer extractor in front of the judge, and
the judge credits the reference answer if the assistant offered it at *any*
point in the conversation, including as a hedged candidate it later dropped.

Two consequences. First, **conversations always run to shard exhaustion**: a
correct answer no longer stops the reveal, so every run costs its full turn
count, and SNOWBALL_V reaches its 21-image peak turn on every conversation
rather than only on the hard ones — serve with at least
`--limit-mm-per-prompt.image 24`. Second, scores read higher than a
classifier-gated, first-correct-answer regime would produce; the two are not
comparable.

RECAP is the exception: it grades only its own recap turn and the reply to it,
never the sharded conversation it replays, or it would be scored for that
run's answer instead of for what the recap recovered.

Verdicts are cached in `logs/layered_vqa/judge_cache.jsonl` (override with
`LAYERED_VQA_JUDGE_CACHE`, choose the grader with `--judge_model` or
`LAYERED_VQA_JUDGE_MODEL`). The key is a hash of the transcript and includes
the judge model, so switching graders re-grades rather than silently reusing
another model's verdicts. Transcripts rarely repeat, so the cache is really
there to make a re-run after a crash free.

The judge quotes the assistant's own words back as `answer_span`, logged as
`exact_answer` on the `answer-evaluation` trace entry — that is what locates
which turn actually answered.

**Grounding, and what a missing `Layers:` line means.** The predicted evidence
set is parsed out of the reply by
`vendor/conv-smi/tasks/layered_vqa/layer_selection.py`. A reply with no readable
`Layers:` line yields `layer_ids: null` and `format_ok: false`, never an empty
set: an empty set is a claim ("no layer supported this"), a missing line is a
format failure, and scoring them alike would report every non-compliant model as
a confidently ungrounded one. `score_experiment_a.py` therefore reports a
format-failure rate alongside the layer metrics, and computes the per-set
metrics over parsed replies only — while joint accuracy is taken over *all*
conversations, since a reply nobody can read did not ground its answer.

The layer manifest drops its position numbers on these samples. `1. layer 0 (the
background layer)` offers two numbers for one image, and a model citing the
position instead of the id would produce a wrong evidence set on every sample.

**Demand levels.** `meta.demand_level` is a heuristic from gold-layer count,
distractor count, dependency-chain length and occlusion. Replace it with a
curated column before making Visual Evidence Cliff claims.

## Scale

300 questions per variant. One model across the eight files is roughly 2,400
conversations; pilot with `--limit` first.
