# Question Sharding and Layer-Evidence Annotation

This pipeline converts the 300 human-authored questions in
`data/derived/mulan-human-review/reviewer_2/accepted_scenes.json` into
question-shard and visual-evidence records for controlled multi-turn VQA
simulations. It adapts the
five validity properties in *LLMs Get Lost in Multi-Turn Conversation*:
information preservation, clear initial intent, order-insensitive follow-ups,
maximal meaningful sharding, and minimal transformation.

## Why short VQA questions need a different threshold

The accepted set contains 93 scenes and 300 questions. The shortest question
has 5 whitespace-delimited words, the median has 16, and the longest has 33.
The source paper filtered out instructions with fewer than three segments, but
that policy would select for unusually verbose VQA questions and encourage
artificial filler turns.

Layered-VQA therefore uses **structured 3-6-shard decomposition**:

- exactly three shards for a short question, separating generic operation,
  expected answer type/property, and target or scope;
- four to six shards when the question contains additional attributes,
  relations, references, count scope, or evidence conditions;
- no image-derived constraint or repeated filler added to reach the minimum;
- all generated shards remain pending human review.

This keeps short questions in the experiment while retaining the
paper's key controlled comparison: all shards together must be informationally
equivalent to the canonical question.

## Generation workflow

Each question uses three structured Responses API calls by default:

1. question-only semantic sharding, with no image and no gold answer;
2. multimodal evidence annotation using the full composite, background,
   retained isolated layers, reviewed captions, question, answer, and shards;
3. independent multimodal verification and correction of the integrated
   annotation.

If the deterministic checks still find a mechanically repairable invariant
violation after step 3, the verifier receives those exact issue labels for one
targeted correction pass. Genuine visual ambiguity remains flagged for human
review rather than being forced into a confident annotation.

The defaults are `gpt-5.6-luna`, high reasoning effort, and high image detail.
Individual RGBA layers are rendered on a synthetic checkerboard at their
original full-canvas size so both light and dark pixels remain visible. The
progressive strip is intentionally omitted because it duplicates the layer
evidence and encodes an unnecessary ordering cue.

The six prompt rubrics are:

- `direct_ask`: requested answer type plus target;
- `attribute_binding`: property, target, and answer-selecting modifiers;
- `spatial`: target, direction, and landmark without relation reversal;
- `occlusion`: occluded/occluding roles and partial-overlap semantics;
- `counting`: count population, scope, and any second requested output;
- `multi_hop_reference`: target-to-intermediate reference chains.

Only the first, question-sharding call is answer-blind. The evidence and final
verification calls receive the human answer because they must annotate the
minimal visual evidence that supports that answer. A deterministic pass checks
the shard invariants plus candidate layer IDs, minimal-set consistency, layer
role coverage, complete many-to-many shard support, dependency validity, and
the exact evidence/distractor partition. Each JSONL line stores source and
prompt hashes, response IDs, token usage, model configuration, validation
issues, confidence, and pending human-review status.

The main per-question outputs are `question_shards`, `gold_layer_ids`,
`layer_roles`, `minimal_sufficient_layer_sets`, `question_shard_support`,
`evidence_dependencies`, and `distractor_layer_ids`.

## Setup and dry run

```powershell
python -m pip install -r requirements-sharding.txt
Copy-Item .env.example .env
```

Add `OPENAI_API_KEY` to `.env`, then validate the dataset and all prompt files
without making an API call:

```powershell
python src\vecgen\generate_question_shards.py --dry-run
```

## Generate all 300 records

```powershell
python src\vecgen\generate_question_shards.py --workers 4
```

The default output is:

```text
data/derived/mulan-human-review/reviewer_2/question_shards.jsonl
data/derived/mulan-human-review/reviewer_2/question_shards.summary.json
data/derived/mulan-human-review/reviewer_2/accepted_scenes_with_shards_and_evidence.json
```

The JSONL file is flushed after every question. Rerunning the same command
skips successful question IDs and retries errors. Use `--overwrite` only when a
fresh run is intended. Useful pilot commands include:

```powershell
# Six category-balanced records with the full three-call workflow.
python src\vecgen\generate_question_shards.py --temp 6 --workers 2

# Five questions from one scene, written to temp-5 outputs.
python src\vecgen\generate_question_shards.py --scene-id lvis_000000006862 --temp 5 --workers 2

# One category only.
python src\vecgen\generate_question_shards.py --category occlusion

# Two-call ablation; not the recommended final corpus.
python src\vecgen\generate_question_shards.py --skip-verification
```

`--temp X` selects a deterministic, category-balanced sample of `X` questions
after any category or scene filters and writes separate `.temp-X` outputs. It
does not contaminate the resumable all-question checkpoint. Omit `--temp` to
process every filtered question.

## Simulation matrix

The primary eight conditions use the same accepted QA record:

| Condition | Question delivery | Visual delivery |
| --- | --- | --- |
| `FULL` | full question | full composite |
| `CONCAT_Q` | all question shards together | full composite |
| `SHARDED_Q` | one question shard per turn | full composite retained |
| `CONCAT_V` | full question retained | all full-canvas layers together |
| `SHARDED_V` | full question retained | one full-canvas layer per turn |
| `SHARDED_VQ` | question constraints accumulate | visual layers accumulate |
| `RECAP` | shard sequence, then full question | shard sequence, then full composite |
| `SNOWBALL` | repeat prior question information and add one shard | repeat prior visual information and add one layer when applicable |

`SHARDED_V` is best framed as a controlled progressive-perception diagnostic,
not as a claim that ordinary users manually transmit semantic layers. It also
models plausible tool-mediated settings: streaming perception, agent-selected
regions/layers, progressive rendering, and bandwidth-limited visual input.

High-value extensions that distinguish the work from a text-only replication:

- `SHUFFLE_Q`: permute all non-intent shards to verify order independence and
  measure recency sensitivity.
- `EVIDENCE_FIRST_V` versus `EVIDENCE_LAST_V`: vary when the minimal gold
  evidence layer arrives while holding the final visual information constant.
- `DISTRACTOR_V`: reveal irrelevant valid layers before evidence layers to
  separate memory load from missing evidence.
- `CAPTION_SHARDED_V`: replace each visual layer with its corrected human
  caption as a modality-control and more realistic tool-message condition.
- `ALIGNED_VQ` versus `MISALIGNED_VQ`: align each textual constraint with its
  supporting layer, or deliberately offset them, to measure cross-modal binding.
- `MISSING_EVIDENCE_V`: withhold one minimal evidence layer as a sanity check
  that success depends on the annotated evidence rather than dataset priors.

Before final experiments, manually accept/edit every shard plan and compare
`FULL` with `CONCAT_Q` and `SHUFFLE_Q`. Material degradation in those controls
indicates information loss or bad decontextualization rather than a genuine
multi-turn effect.
