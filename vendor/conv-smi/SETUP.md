# Setup & running

Everything needed to go from a fresh checkout to logged conversations, for
this directory only. `README.md` explains what the code does;
`../../docs/lic_simulation.md` maps the experiment conditions (FULL, CONCAT_V,
SHARDED_VQ, RECAP, SNOWBALL, EVIDENCE_*) onto these commands.

> [!IMPORTANT]
> **Every command runs from `vendor/conv-smi/`.** Prompts, the judge cache and
> `logs/` are all opened by relative path. From anywhere else the first file
> open fails:
> ```
> FileNotFoundError: '../../data/derived/lic/layered_vqa_text.json'
> ```

---

## Quickstart

```sh
cd vendor/conv-smi
pip install -r requirements.txt

# pick ONE backend
export VLLM_BASE_URL=http://localhost:8000/v1     # local, free
export OPENROUTER_API_KEY=sk-or-...               # closed models, billed

export LAYERED_VQA_DATASET=../../data/derived/lic/layered_vqa_text.json

python model_openai.py --model local-vlm          # does the wiring work?
python simulator_full.py --assistant_model local-vlm --verbose   # 1 sample, unlogged

python run_simulations.py \
  --dataset_file ../../data/derived/lic/layered_vqa_text.json \
  --models local-vlm --judge_model local-vlm --user_model local-vlm \
  --N_full_runs 1 --N_concat_runs 1 --N_sharded_runs 1 --N_workers 4
```

---

## 1. Install

```sh
pip install -r requirements.txt
```

Python 3.13 is what this checkout runs on. The code uses `X | None`
annotations throughout, so 3.10 is the realistic floor.

| Package | Needed for |
|---|---|
| `openai` | every model call (all backends speak this API) |
| `PyYAML` | reading `models.yaml` and `agents.yaml` |
| `GitPython` | stamps the current commit into every log record |
| `pymongo` | `bson.ObjectId`, the `conv_id` on each record |
| `pandas`, `tqdm` | progress bars and log summaries |
| `streamlit` | the conversation viewer only — skip it if you never open the UI |

`numpy` and `tiktoken` are listed but nothing imports them.

**You must be inside a git repository.** `log_conversation` records
`git.Repo(search_parent_directories=True).head.object.hexsha`, so logging
raises outside a checkout.

## 2. Point at a model

Every model — local or hosted — is reached through the OpenAI
chat-completions API. [`models.yaml`](models.yaml) is the only thing that
decides which endpoint serves which name:

```yaml
models:
  local-vlm:         {backend: vllm,       model: local-vlm}
  claude-sonnet-4.5: {backend: openrouter, model: anthropic/claude-sonnet-4.5}
```

The keys are what you pass to `--models`, `--user_model` and `--judge_model`,
and what is written into the logs. Add a model by editing the file, or pass
`backend:id` inline for a one-off (`--judge_model openrouter:openai/gpt-4o-mini`).

### Open weights, locally (free)

Serve with enough image slots for the widest *conversation*, not the widest
turn: SNOWBALL_V re-attaches every revealed layer each turn, so a 6-layer
scene peaks at 21 images in one prompt.

```sh
vllm serve <vision-model> --served-model-name local-vlm --limit-mm-per-prompt.image 24
export VLLM_BASE_URL=http://localhost:8000/v1
```

For Qwen3.5 in non-thinking mode, set the chat-template default on the vLLM
server; an answer-only prompt controls visible style but does not itself disable
the model's internal thinking mode:

```sh
vllm serve Qwen/Qwen3.5-9B \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --limit-mm-per-prompt.image 24
```

Qwen3.5 does not use the older `/nothink` soft switch. With
`enable_thinking: false`, the assistant returns only its direct response; the
judge's JSON `reasoning` remains the judge's own grading rationale and does not
depend on assistant reasoning.

`--served-model-name` must match a `model:` in `models.yaml`. `VLLM_BASE_URL`
overrides the `base_url` there. Set `VLLM_API_KEY` only if you started vLLM
with `--api-key`.

### Closed weights, through OpenRouter (billed)

```sh
export OPENROUTER_API_KEY=sk-or-...
```

Clients are built lazily per backend, so a purely local run never needs this
key. Mixing is normal and sensible — assistant on OpenRouter, judge local, so
the cheap role stays free.

> [!WARNING]
> Nothing here tracks spend. There is no pricing table and no `cost_usd` on
> any record — check OpenRouter's dashboard.

Every run prints where each model's calls go. Read these lines: the same
command that used to hit a free local server can start billing an account.

```
[models] local-vlm -> vllm:local-vlm
[models] gpt-4o-mini -> openrouter:openai/gpt-4o-mini
```

## 3. Tune generation settings (optional)

[`agents.yaml`](agents.yaml) holds temperature, token budgets, timeouts and
retries for the three agents that make model calls — `assistant` (the model
under test), `user` (the simulated user revealing shards) and `judge`. Editing
a number there changes every call that agent makes. The shipped profile keeps
temperature and token caps identical across roles and gives every simulator the
same budget: temperature `1.0`, regular cap `1024`, and reasoning-capable cap
`16384`. A reasoning cap is only a completion limit; with thinking disabled the
model emits a direct answer and no reasoning trace is required.
The settings in force are announced at startup next to the routing lines:

```
[agents] assistant/sharded -> temperature=1.0 max_tokens=1024 timeout=600s retries=3
[agents] judge -> temperature=1.0 max_tokens=1024 timeout=600s retries=3
```

Print them without running anything:

```sh
python agent_config.py --model local-vlm
```

> [!IMPORTANT]
> Temperature is part of a run's identity. Any assistant or user temperature
> other than 1.0 adds a suffix to the `conv_type` (`full-t0.7`,
> `sharded-at0.7-ut1.0`), so those runs get their own log folder and never mix
> with the 1.0 ones — which also means changing a temperature starts a fresh
> set of runs instead of resuming the existing ones.

## 4. Pick a dataset

Samples are exported by the parent repo, not by anything here:

```sh
cd ../.. && python src/vecgen/export_lic_dataset.py --all && cd vendor/conv-smi
```

That writes eight variants to `data/derived/lic/` (already present in this
checkout). Use `--limit 5` there to build a small file to pilot against —
the simulators have no sampling flag of their own.

| File | Serves |
|---|---|
| `layered_vqa_text.json` | FULL (Exp-A 1), CONCAT_Q, SHARDED_Q, RECAP_Q, SNOWBALL_Q, SHUFFLE_Q-CONCAT |
| `layered_vqa_grounding.json` | Exp-A 2: full image + all layers, answer **and** layer ids |
| `layered_vqa_oracle_gold.json` | Exp-A 3: only a gold minimal sufficient layer set |
| `layered_vqa_visual_canonical.json` | CONCAT_V, SHARDED_V, SNOWBALL_V, RECAP_V |
| `layered_vqa_aligned_vq.json` / `layered_vqa_misaligned_vq.json` | SHARDED_VQ |
| `layered_vqa_visual_evidence_first.json` / `..._last.json` | EVIDENCE_FIRST_V / EVIDENCE_LAST_V |

The two Experiment A files are single-turn: one shard holding the whole
question, every image attached to it, so `--N_full_runs 1` is the whole
condition. Only `layered_vqa_grounding.json` asks for layer ids, and only its
runs get layer metrics.

> [!CAUTION]
> **Selection has two independent paths — set both, to the same file.**
> `run_simulations.py --dataset_file` loads the samples for that launch;
> everything else (`simulator_recap.py`, `simulator_snowball.py`, the viewer)
> reads `$LAYERED_VQA_DATASET`. A mismatch produces runs that look fine and
> answer the wrong condition.

```sh
export LAYERED_VQA_DATASET=../../data/derived/lic/layered_vqa_text.json
```

There is no `.env` loading. `utils.load_env_vars` exists but nothing calls it
— export the variables yourself.

## 5. Verify before spending

Three checks, cheapest first.

```sh
python agent_config.py            # offline: config parses, settings resolve
python model_openai.py --model local-vlm    # one call: routing, auth, transport
python simulator_full.py --assistant_model local-vlm --verbose   # 1 random sample, nothing logged
```

The third prints the full exchange and the judge verdict without writing a log
record — the fastest way to see prompts, images and grading end to end.

## 6. Run

| Entry point | Conditions |
|---|---|
| `run_simulations.py` | full / concat / shuffle-concat / sharded |
| `simulator_snowball.py` | snowball (own CLI, unreachable from `run_simulations.py`) |
| `simulator_recap.py` | recap — *replays finished sharded logs*, so run sharded first |
| `simulator_full.py`, `simulator_sharded.py` | one random sample, for eyeballing |

### The main driver

```sh
python run_simulations.py \
  --dataset_file ../../data/derived/lic/layered_vqa_text.json \
  --tasks layered_vqa \
  --models local-vlm --judge_model local-vlm --user_model local-vlm \
  --N_full_runs 1 --N_concat_runs 1 --N_sharded_runs 1 \
  --N_shuffle_concat_runs 0 \
  --N_workers 4
```

Each `--N_*_runs` is a target count per sample, not a delta: set the ones you
don't want to `0`. `--models` takes several names and runs each.

### Who reveals the shards: `--user_agent`

- `scripted` — walks the exported shard order. Required whenever reveal order
  *is* the manipulation.
- `scripted-shuffle` — intent shard first, the rest in a fresh random order.
- `llm` — the user model improvises the order. The historical default, and
  still the default on text-only datasets.

On any dataset carrying per-shard images the drivers **refuse to default it**
and make you choose, because reveal order is part of the design:

```
error: this dataset carries per-shard images, so reveal order is part of the
design: pass --user_agent explicitly
```

An explicit `--user_agent llm` there is allowed as its own arm — it prints a
`[warn]` and logs under the llm `conv_type`, so the two never mix.

```sh
# SHARDED_V — canonical arrival order
LAYERED_VQA_DATASET=../../data/derived/lic/layered_vqa_visual_canonical.json \
python run_simulations.py \
  --dataset_file ../../data/derived/lic/layered_vqa_visual_canonical.json \
  --tasks layered_vqa --models local-vlm \
  --judge_model local-vlm --user_model local-vlm \
  --user_agent scripted \
  --N_full_runs 0 --N_concat_runs 0 --N_sharded_runs 1
```

### Snowball

Reads samples through the task, so `LAYERED_VQA_DATASET` is the only thing
selecting the variant — there is no `--dataset_file`.

```sh
LAYERED_VQA_DATASET=../../data/derived/lic/layered_vqa_text.json \
python simulator_snowball.py --task layered_vqa \
  --models local-vlm --judge_model local-vlm --user_model local-vlm \
  --N_snowball_runs 1 --N_workers 4
```

`--models` is the flag that takes effect. `--assistant_model` is accepted here
but never read.

### Recap

Replays finished sharded conversations, so the matching sharded run must exist
first, pointed at the same dataset. It picks up logs whose `dataset_fn`
matches, scanning `sharded` and `sharded-scripted` (override with
`--sharded_conv_types`), and skips what it has already recapped.

```sh
LAYERED_VQA_DATASET=../../data/derived/lic/layered_vqa_text.json \
python simulator_recap.py --task layered_vqa --models local-vlm
```

Here `--models` is a *filter* over which logs to replay — the assistant model
comes from the replayed record, not from this flag.

### Every condition, one model at a time

The 14 Experiment-B conditions plus Experiment A and the per-model report are
scripted, so a full launch is one command. The two runners differ only in where
the model lives; both take their condition list from `lic_conditions.sh`.

```sh
bash run_all_open_models.sh                        # local vLLM, one GPU
bash run_all_closed_models.sh                      # OpenRouter, billed
bash run_all_closed_models.sh --only gpt-5.6-luna  # one of the three
bash run_all_closed_models.sh --dry-run            # print, call nothing
bash run_all_closed_models.sh --log_folder logs/Final-Closed-1 --only minimax-m3
```

One model per launch into a shared `--log_folder` is the normal shape for a
repeated run: each writes its own `results_<model>.md` and
`primary_analysis_<model>.md`, and the folder's judge cache is shared, so three
launches into `logs/Final-Closed-0` leave three of each rather than overwriting
one.

`run_all_closed_models.sh` reads `OPENROUTER_API_KEY` from the environment, and
falls back to the repo's `.env` (by grep, not by sourcing it). It refuses any
alias that `models.yaml` routes somewhere other than OpenRouter, so an
open-weight name cannot quietly become a bill.

After each model finishes, both runners re-grade **that model's own logs** for
the behavioural measures the simulators do not record — which turn answered,
what had been revealed by then, every answer attempt, verbosity, layer
citations — and write the tables to `<log_folder>/primary_analysis_<model>.md`
plus per-table CSVs, with the graded records under
`<log_folder>/primary-analysis/<model>/`. That subdirectory name is deliberate:
`posthoc/` stays reserved for the cross-judge matrix, so the two never mix in
one folder. That is one further judge call per conversation, so `--no-posthoc` leaves
it out. It is pinned to `--assistants <model> --judges <model>`: the diagonal
only, never the judge x assistant matrix, which is a separate pass
(`run_posthoc_judging.sh`) and costs N² for N models.

### Smoke test: every condition, N questions

Before a paid launch, run the same conditions on a handful of samples. This is
a plumbing check — routing, auth, both user agents, image materialization, the
judge, resume accounting, the scorer. At `--n 1` it says nothing about accuracy.

```sh
bash smoke_test_closed_models.sh --n 1                       # all three models
bash smoke_test_closed_models.sh --n 1 --only gpt-5.6-luna   # one of them
bash smoke_test_closed_models.sh --n 2 --only minimax-m3,gemini-3.7-flash
bash smoke_test_closed_models.sh --n 1 --dry-run             # print, call nothing
```

There is no `--limit` on the simulators — a second way to score a subset
silently is not wanted — so the size is cut in the data instead:
`make_smoke_datasets.py` writes N-sample copies of all eight dataset files into
`../../data/derived/lic/.smoke-n<N>/`, keeping their original basenames because
the logs and both scorers key conditions off those, and prefixing every relative
image path with `../` for the extra directory level. That directory ignores
itself in git.

Conditions run to the end even when one fails, and the summary is one PASS/FAIL
line each; the script exits non-zero if any failed. Logs go to
`logs/Smoke-Closed-n<N>/` with their own judge cache, so they never mix into a
real run.

## 7. What you get

```
logs/layered_vqa/<conv_type>/<conv_type>_layered_vqa_<model>.jsonl
logs/layered_vqa/judge_cache.jsonl
```

One JSON record per conversation: the full trace, `is_correct`, `score`,
`judge_model`, and the git commit. `logs/` is gitignored.

**Runs resume.** Every launch counts what is already logged for that
`conv_type` and only runs the shortfall, so re-running the same command after
a crash costs only what was missing. This is also why `conv_type` carries the
temperature and user-agent suffixes — a run shaped differently must not be
counted as one of these.

**A judge outage is a failed run, not a zero.** Grading raises before any log
write, so the next launch redoes that conversation instead of freezing a false
`is_correct: false` into the results.

Experiment A's grounding runs additionally carry `layer_selection` and
`layer_metrics` inside each conversation's `answer-evaluation` trace entry.
Turn the finished logs into its metrics table with:

```sh
python score_experiment_a.py               # markdown table
python score_experiment_a.py --csv exp_a.csv   # one row per conversation
```

Experiment B is scored the same way, one row per delivery condition plus the
paired contrasts, and a **coverage** column — a conversation that overflowed the
server's context window is absent from the logs rather than scored wrong, so an
accuracy at coverage < 1.000 is an accuracy over a subset:

```sh
python score_experiment_b.py --log_folder logs/Layered-VQA-0
python score_experiment_b.py --csv exp_b.csv   # one row per conversation
```

For one file per model holding both experiments — what
`run_all_open_models.sh` writes as each model finishes:

```sh
python score_model_report.py --log_folder logs/Layered-VQA-0 --models qwen3.5-9b
# -> logs/Layered-VQA-0/results_qwen3.5-9b.md
```

## 8. Inspect

```sh
pip install streamlit
streamlit run app_conv_viewer.py
```

Reads `logs/` and `$LAYERED_VQA_DATASET`; the sidebar lists whatever
`conv_type` folders exist.

## 9. Tests

They live in the parent repo and run from the repo root (they `chdir` in
themselves):

```sh
cd ../.. && python3 -m pytest tests/test_layered_vqa_simulation.py tests/test_conversation_simulator_multimodal.py
```

The OpenAI client is stubbed at the transport boundary, so no key or server is
needed and nothing is billed. Run them after touching `multimodal.py`,
`model_openai.py`, or any simulator.

## 10. Environment variables

| Variable | Effect |
|---|---|
| `LAYERED_VQA_DATASET` | which exported variant the *task* loads (recap, snowball, viewer) |
| `VLLM_BASE_URL` | vLLM endpoint; overrides `base_url` in `models.yaml` |
| `VLLM_API_KEY` | only if vLLM was started with `--api-key` |
| `OPENROUTER_API_KEY` | required for any OpenRouter model |
| `LAYERED_VQA_JUDGE_MODEL` | grader; `--judge_model` sets this for you |
| `LAYERED_VQA_JUDGE_CACHE` | verdict cache path (default `logs/layered_vqa/judge_cache.jsonl`) |
| `LAYERED_VQA_MODELS_CONFIG` | alternative `models.yaml` |
| `LAYERED_VQA_AGENTS_CONFIG` | alternative `agents.yaml` |

## 11. Troubleshooting

| Symptom | Cause |
|---|---|
| `FileNotFoundError: '../../data/derived/lic/...'` | not in `vendor/conv-smi/`, or the dataset was never exported |
| `ValueError: Unknown model 'x'` | not in `models.yaml`; add it or pass `backend:id`. Never guessed — a guess is either a confusing connection failure or an unintended bill |
| `ValueError: Backend 'openrouter' needs an API key` | `OPENROUTER_API_KEY` unset |
| `Failed to get a chat completion ... after 3 attempt(s)` | server down or model name not served. 4xx other than 408/429 fails immediately — the request itself is wrong, so resending cannot help |
| `error: ... pass --user_agent explicitly` | image-carrying dataset; choose `scripted` or `llm` |
| Every launch re-runs everything | the `conv_type` moved — a temperature change, a different `--user_agent`, or a different `--log_folder` |
| `aborting <task_id>: N user turns without revealing every shard` | the LLM user agent is stuck; the run is deliberately left unlogged so an under-revealed conversation never lands in the results |
| Model sees near-black images | raw RGBA layers were used instead of the flattened renders — never point the simulator at the source layer PNGs |
| `ModuleNotFoundError: streamlit` | viewer-only dependency; `pip install streamlit` |
