# CLAUDE.md — vendor/conv-smi

Multi-turn conversation simulator, vendored (plain directory, not a submodule)
into the Layered-VQA repo and trimmed to a single task, `layered_vqa`. Edit the
files here directly; there is no upstream sync step.

`SETUP.md` is the install-and-run guide (env vars, entry points, troubleshooting
— keep its commands in sync when a flag changes).
`README.md` documents the code. `../../docs/lic_simulation.md` documents how the
experiment conditions (FULL, CONCAT_Q/V, SHARDED_Q/V/VQ, RECAP, SNOWBALL,
EVIDENCE_FIRST/LAST_V, …) map onto these entry points — read it before changing
anything that affects what a condition delivers.

## Always run from this directory

Prompts, the judge prompt, the judge cache, and `logs/` are all opened by
relative path (`prompts/user_agent.txt`, `prompts/layered_vqa/*`,
`logs/layered_vqa/judge_cache.jsonl`). Running from the repo root silently fails
at the first file open. Tests handle this by `chdir`-ing in.

## Entry points

| Script | Conditions |
|---|---|
| `run_simulations.py` | full / concat / shuffle-concat / sharded |
| `simulator_snowball.py` | snowball (own CLI; not reachable from `run_simulations.py`) |
| `simulator_recap.py` | recap — *replays finished sharded logs*, so the sharded run must exist first |
| `score_experiment_a.py` | aggregates the finished Experiment A logs into its metrics table |
| `score_experiment_b.py` | the same for the 14 Experiment B delivery conditions |
| `score_model_report.py` | both experiments for one model, written to a markdown file |
| `posthoc_judge.py` | re-grades finished logs with any judge; adds attempt-level and grounding fields |
| `posthoc_analysis.py` | turns those into the seven behavioural tables |
| `app_conv_viewer.py` | `streamlit run app_conv_viewer.py` |
| `sample_judge_validation.py` | draws the conversations people grade; the review UI is `vendor/judge-review` |

All three Experiment A conditions are single-turn `full` runs, separated only by
`dataset_fn` (`layered_vqa_text` / `_grounding` / `_oracle_gold`), the same way
CONCAT_V already is. `get_run_counts` filters on `dataset_fn`, so they resume
independently without any new conv_type.

**An Experiment-B condition is a (conv_type, dataset_fn) pair, not a folder.**
`sharded-scripted` alone holds SHARDED_V, both SHARDED_VQ arms and both
EVIDENCE_* arms; `recap-concat` holds RECAP_Q and RECAP_V. `CONDITIONS` in
`score_experiment_b.py` is that mapping, and it also matches the
temperature-suffixed folder names `utils_log.temperature_suffix` produces, so a
non-baseline run is scored rather than silently missed.

`score_experiment_b.py` reports **coverage** (logged conversations / dataset
size) next to every accuracy. A conversation whose prompt overflows the server's
context window raises instead of logging a verdict, so the sample vanishes from
the file rather than scoring zero — without coverage, an arm that dropped its
longest conversations reads as an arm that did well on them. That is also why
`run_all_open_models.sh` sizes `--max-model-len` from the checkpoint's own config
(64k target) instead of a fixed 32768.

`run_all_open_models.sh` calls `score_model_report.py` after each model, while
that model is still loaded, writing `logs/<folder>/results_<alias>.md` with both
experiments in it.

**The condition list lives in `lic_conditions.sh`, sourced by
`run_all_closed_models.sh` and `smoke_test_closed_models.sh`.** It is the one
copy of "what the 14 Experiment-B conditions and the 3 Experiment-A conditions
actually are"; a runner that spelled a condition its own way would produce logs
nothing else could score. (`run_all_open_models.sh` still carries its own copy
— it predates the split; move it over rather than editing both if it needs a
change.) Closed models need no GPU, so that runner is the same conditions minus
download/serve/unload, plus a refusal to run any alias `models.yaml` does not
route to OpenRouter.

`smoke_test_closed_models.sh --n N` runs every condition on N samples. There is
deliberately no `--limit` flag on the simulators, so the cut happens in the
data: `make_smoke_datasets.py` writes N-sample copies into
`data/derived/lic/.smoke-n<N>/` under the **original basenames** (logs and both
scorers key conditions off the basename) with image paths prefixed by `../` for
the extra directory level. It keeps going past a failing condition and reports
one PASS/FAIL line each, because knowing only the first broken condition is
worth little.

## Model routing lives in `models.yaml`

Every model speaks the OpenAI chat-completions API. Open weights are served
locally by vLLM; closed weights go through OpenRouter, which speaks the same
API. `models.yaml` is the only thing that decides which
is which, and `model_registry.py` reads it — adding a model is a config edit,
not a code change.

The keys under `models:` are what you pass to `--models` / `--judge_model` /
`--user_model` and what lands in the logs; `model:` is the id sent on the wire,
so an upstream rename does not invalidate a run's logs. `backend:id` works
inline for a model not yet in the config (`--judge_model
openrouter:openai/gpt-4o-mini`).

**An unknown model name is an error, never a guess.** Guessing means either a
confusing connection failure or an unintended bill. For the same reason the
registry prints one `[models] <alias> -> <backend>:<id>` line per model: a
command that used to hit a free local server can now charge an account, and
that should never be silent.

Needs `VLLM_BASE_URL` (or the `base_url` fallback in the config) for vLLM, and
`OPENROUTER_API_KEY` for OpenRouter. Clients are built lazily per backend, so a
purely local run never needs an OpenRouter key.

## Generation settings live in `agents.yaml`

`models.yaml` says *where* a call goes; `agents.yaml` says *how* it is made.
Three agents make every model call — `assistant` (the model under test), `user`
(the simulated user; the scripted agent makes no calls, so nothing there
reaches it) and `judge` — and each gets `temperature`, `top_p`, `top_k`,
`max_tokens`, `reasoning_max_tokens`, `timeout`, `max_retries`, plus optional
`per_simulator:` overrides keyed by `full` / `sharded` / `snowball` / `recap`.
The shipped config deliberately has no overrides: all four simulators resolve
the same profile. `reasoning_max_tokens` is a completion cap, not a request for
visible reasoning; with thinking disabled it caps the direct response.
`top_p` and `top_k` are sent only when non-null, and `top_k` rides in
`extra_body` because it is not in the OpenAI schema. It ships as `0`, which
current vLLM and OpenRouter both read as *disabled*, so the shipped profile
truncates nothing and samples as it did before the knob existed (vLLM also
accepts its legacy `-1`; vLLM before ~0.8.5 rejects `0` with a 400). Neither
knob reaches the conv_type, so unlike temperature they can silently mix sampling profiles inside
one log folder. `agent_config.py` reads it; call sites do
`agent_config.assistant("sharded").call_kwargs(model, temperature=...)` and
splat the result into `generate`. **No simulator should carry a generation
literal again** — that is how the same knob ended up with three different
values depending on the entry point.

Unknown keys are an error, for the same reason an unknown model is: a
misspelled `max_tokens` that "worked" would run a whole experiment on the
provider's default budget. Settings are announced once per agent
(`[agents] assistant/sharded -> temperature=1.0 …`) because temperature decides
what a results table means and is not recoverable from the logs.

Precedence is CLI > `agents.yaml`. `--assistant_temperature` /
`--user_temperature` default to `None`, meaning "read the file"; the
simulators' `temperature=None` constructor defaults mean the same. Don't
restore a numeric default to either — a literal `1.0` there would silently
outrank the config.

`LAYERED_VQA_AGENTS_CONFIG` points elsewhere. A missing file at *that* path is
an error; a missing default `agents.yaml` falls back to `BUILTIN`, which
reproduces the shipped profile.

## Dataset selection has two independent paths

- `run_simulations.py --dataset_file` loads the samples for that launch.
- `TaskLayeredVQA.get_dataset_file()` reads `$LAYERED_VQA_DATASET` (default
  `../../data/derived/lic/layered_vqa_text.json`).

`simulator_recap.py`, `simulator_snowball.py` and the viewer go through the
*task*, so they only see `LAYERED_VQA_DATASET`. Set both, to the same file. A
mismatch produces runs that look fine and answer the wrong condition.

Other env vars: `LAYERED_VQA_JUDGE_MODEL`, `LAYERED_VQA_JUDGE_CACHE`. Every
entry point also takes `--judge_model`, which just sets
`LAYERED_VQA_JUDGE_MODEL` before the task is built — `AnswerJudge` reads it
once, at construction, so `apply_judge_model(args)` has to run before the first
`get_task()`.

## Invariants worth protecting

**`conv_type` strings must match between writer and resumer.** Logging
(`log_conversation`) and resume accounting (`get_run_counts`) build the same
string independently, and if they diverge by one character no completed run is
ever counted and every launch re-runs everything at full cost. The temperature
half of that string now comes from one function, `utils_log.temperature_suffix`
— both sides call it, so keep it that way rather than re-deriving the format.
The user-agent half is still built per driver (`simulator_sharded`,
`snowball_conv_type`, `recap_conv_type`). Adding a run-shaping flag means
updating both sides.

Temperature reaches those strings from `agents.yaml` as well as from the flags,
so `run_simulations.py` resolves it *per simulator*
(`assistant_temp("full")` / `assistant_temp("sharded")`): a `per_simulator:`
override of the assistant temperature would otherwise make one of the two
conv_types wrong. `simulator_recap` matches its recap kind with
`startswith("recap-full")`, not `==`, because the conv_type may carry a
temperature suffix.

**Image-carrying datasets require an explicit `--user_agent`.** On those
datasets reveal order is part of the design, so `resolve_user_agent`
(`user_agent_scripted.py`, called by every driver before todos are built)
refuses the implicit default: `scripted` delivers the exported order (the
canonical / EVIDENCE_* / aligned / misaligned conditions), and an *explicit*
`llm` is the improvised-order arm — allowed deliberately, announced with a
`[warn]` line, and logged under the llm conv_type (`sharded` / `snowball`) so
the two arms never mix. Text-only datasets keep the historical `llm` default.
`ScriptedUserAgent` walks `sample["shards"]` in exported order;
`scripted-shuffle` keeps the intent shard first and randomizes the rest. The
LLM agent's shard lists advertise attachments by their neutral labels
(`"attaches": ["layer 2"]`) — labels only, never filenames.

**Shard ids from the LLM user agent are untrusted.** The loop guards
(`len(revealed) >= len(shards)`) count reveal log entries, and the id feeding
them comes out of model JSON — so `UserAgent` coerces it to `int` and maps
anything that is not a fresh, valid shard id to -1 before it reaches a
simulator. Both multi-turn loops also carry a `max_user_turns` cap
(2·shards+6) that aborts a stuck conversation with an exception rather than
logging an under-revealed run as complete.

**The prompt a sample gets is chosen by `meta.response_format`.** `grounding`
samples carry `answer_and_layers`, which serves
`prompts/layered_vqa/layered_vqa_grounding_prompt.txt` (answer + `Layers:` +
confidence + why) instead of the plain answer prompt; every other variant asks
for the answer alone. Two things ride on that switch and must move together:
the manifest **drops its position numbers** for those samples (a list reading
`1. layer 0 (the background layer)` offers two numbers for one image, and a
model citing the position produces a wrong evidence set on every sample), and
`evaluate_conversation` scores the predicted set.

**Grounding is scored inside the evaluation dict, not by a simulator.**
`tasks/layered_vqa/layer_selection.py` parses the reply and scores it against
`gold_layer_ids` / `minimal_sufficient_layer_sets` / `distractor_layer_ids`;
`evaluate_conversation` merges `layer_selection`, `layer_metrics`,
`joint_correct` and `joint_complete` into what it returns, and
`grade_conversation` already writes that whole dict into the trace's
`answer-evaluation` entry. That is why no simulator knows about Experiment A —
keep it that way. The metrics are deterministic and recomputed every run, so the
judge cache (which holds only the answer verdict) never staleness-poisons them.

A reply with no readable `Layers:` line yields `layer_ids: None` /
`format_ok: False`, **never an empty list**. An empty set is a claim; a missing
line is a format failure, and collapsing the two would report every
non-compliant model as a confidently ungrounded one.

**Never leak image paths into agent prompts.** Shard dicts carry an `images`
key whose filenames (`.../layer_2.png`) name the answer. `user_agent.py`
projects shards down to `{shard_id, shard}` before prompting and renders its
conversation view with `extract_conversation(..., hide_image_paths=True)`,
which emits a bare `[image]` per image part; `render_transcript`
(`tasks/layered_vqa/task_layered_vqa.py`) delegates to the same call for the
judge transcript. Any new code that serializes shards or a trace into a
prompt must go through that flag — plain `content_to_text` emits
`[image: <path>]` and is only for logs and terminal output.

**Logs keep readable image paths; only the API call sees base64.**
`materialize_local_images` converts local paths to data URLs inside
`model_openai.generate`, on a deep copy. Paths resolve relative to the *dataset
JSON file's* directory (`image_base_dir`), not the CWD. Don't move that
conversion upstream — it would bloat every log record.

**Grading is once per conversation, over the whole transcript.** There is no
turn classifier or separate pre-judge extractor: a conversation runs to shard
exhaustion, then `utils.grade_conversation` makes one `evaluate_conversation`
call and appends a single `answer-evaluation` entry. The judge is told to
credit the reference answer if the assistant offered it at *any* point, so
conversations never terminate early — a correct answer mid-run does not stop
the reveal. All four simulators go through `grade_conversation`; keeping that
one copy is what stops their `is_correct` rules drifting apart again.

`evaluate_conversation` returns a dict with `{"score": float}` and/or
`{"is_correct": bool}`; `grade_conversation` asserts on it. The judge returns
`reasoning`, `answer_span`, and `is_correct`. `reasoning` is the judge's own
short grading rationale, not assistant chain-of-thought. `answer_span` is the
assistant's own words quoted back by the judge; it is logged as `exact_answer`
and is how analysis locates the turn that answered.

A judge outage is a failed run, not a zero: `grade_conversation` raises
`JudgeUnavailableError` when the judge returns its error verdict, before any
log write, so the next launch redoes the conversation instead of freezing
`is_correct: false` into the results forever.

Verdicts are still cached on disk (`JudgeCache`, thread-safe, one instance
per cache file shared process-wide via `shared_judge_cache` — never construct
`JudgeCache` directly), keyed by a hash of the whole transcript. Two
multi-turn conversations rarely match, so hits come mostly from repeated
single-turn runs whose short replies coincide; the cache also makes a re-run
after a crash free. The key includes the judge model, so swapping graders
re-grades rather than reusing another model's verdicts.

**RECAP grades only its own exchange.** `simulator_recap` passes
`graded_turns=recap_trace[-2:]`. It replays a finished sharded conversation,
and since sharded runs no longer stop when they answer correctly, a judge shown
the full trace would credit the recap for the original run's answer.

**Nothing tracks cost.** There is no pricing table, no `cost_usd` on trace
entries, and no per-call accounting; `generate` returns token counts only.
OpenRouter's dashboard is the place to look at spend.

## Tests

Live in the parent repo, run from the repo root:

```sh
python3 -m pytest tests/test_layered_vqa_simulation.py tests/test_conversation_simulator_multimodal.py
```

They stub the OpenAI client at the transport boundary (not at
`model_openai.generate`), so prompt population, data-URL materialization, and
turn bookkeeping are all really exercised. Run them after touching
`multimodal.py`, `model_openai.py`, or any simulator.

## Adding a task

Implement the `task_base.Task` abstract methods, put the code in
`tasks/<name>/`, and register it in `tasks/tasks.py`. Note `TaskLayeredVQA`
does not accept the `version` kwarg `get_task` can forward, and doesn't call
`super().__init__`.


## Post-hoc analysis

`run_posthoc_judging.sh` serves each model in turn and re-grades every
conversation in `logs/Final-{0,1,2}`, so the full 6x6 judge-by-assistant matrix
exists rather than only the diagonal the simulators produced. Grading is
**text-only**: the judge reads a transcript whose image parts are a bare
`[image]`, which is why that script serves 32k and no `--limit-mm-per-prompt`.

Two things ride along in the same judge call, at no extra cost:

* **Answer attempts.** The judge lists *every* attempt, right or wrong. The
  position of the first one, in fifths of the conversation, is the premature
  commitment measure. A missed early *wrong* attempt moves that position, which
  is why the prompt spends a rule on it.
* **Turn location.** `first_correct_turn` plus a verbatim `answer_span` that this
  code independently locates among the turns. `turn_conflict` counts the
  disagreements, so a judge that cannot cite itself is detectable rather than
  silently trusted.

Everything else is computed without model calls, from the reveal log
(`shard_revealed` for sharded/recap, `hint_revealed` for snowball) and the
dataset annotation: verbosity quintiles, answer-bloat curves, the layer-citation
matrix, and `ungrounded_correct`.

**`ungrounded_correct` is the one that changes the headline numbers.** It marks
a correct answer given at a turn where a gold layer had not yet been delivered.
Plain accuracy credits that identically to an answer read off the evidence; the
dataset's `gold_layer_ids` plus the reveal timing are what make them separable
at all. Report grounded accuracy alongside accuracy, not instead of it.

Errored gradings go to `posthoc/<judge>/errors.jsonl`, never to the results
file, so the next launch retries them instead of freezing a false negative.

**The closed-model runners fold this pass in per model** (`run_posthoc` in
`lic_conditions.sh`, after `write_model_report`), pinned to
`--assistants <model> --judges <model>`. Everything except the cross-judge
table therefore lands in a primary run: first-attempt position, verbosity
quintiles, answer bloat, the citation matrix, `ungrounded_correct` and answer
churn. The matrix itself stays out on purpose — it is N² passes for N models,
which is a bill rather than an afternoon on metered endpoints, so
`posthoc_analysis.py`'s cross-judge table comes out 1x1 there and is the one
output to ignore. `--no-posthoc` / `SKIP_POSTHOC=1` skips it; `--data_dir` is
passed as `$D` so a smoke run reads the same cut-down datasets its
conversations came from.

Both scripts take **`--subdir`** (default `posthoc`, so every existing
`logs/Final-*/posthoc/` tree and `run_posthoc_judging.sh` are untouched). The
closed-model runners pass `primary-analysis`: a measure produced by the primary
run should not be filed under a name asserting it came afterwards, and keeping
the trees apart is what lets one log folder hold both this diagonal and a later
cross-judge pass without either reading the other's records.
