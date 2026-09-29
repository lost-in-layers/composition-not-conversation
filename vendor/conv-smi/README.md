# Conversation simulator

The multi-turn simulation harness Layered-VQA runs its conditions through:
single-turn (FULL, CONCAT), multi-turn (SHARDED), and the recap and snowball
interventions. `tasks/layered_vqa/` is the only task defined here.

**New here? Start with [SETUP.md](SETUP.md)** — install, model routing, env
vars, and copy-paste commands for every entry point.

See `docs/lic_simulation.md` in the parent repository for how the conditions are
actually driven; this file only documents the code in this directory.

# Table of Contents

- [Repository Contents](#repository-contents)
- [Simulating Conversations](#1-simulating-conversations)
- [Viewing Simulated Conversations](#2-viewing-simulated-conversations)
- [Adding a New Task](#3-adding-a-new-task)
- [Dataset Contents](#dataset-contents)

## Repository Contents

- `run_simulations.py`, which drives FULL / CONCAT / SHARDED runs.
- Simulator code (`simulator_*.py`) for single-turn and multi-turn conversations,
  plus the recap and snowball interventions.
- `score_experiment_a.py`, which aggregates the finished single-turn logs into
  the Experiment A metrics table (answer accuracy, layer precision/recall/F1,
  exact evidence match, minimal-set success, joint accuracy).
- `score_experiment_b.py`, which does the same for the 14 Experiment B delivery
  conditions (answer accuracy, conversation length, coverage, and the paired
  contrasts: sharding cost, recap recovery, alignment cost, evidence order).
- `score_model_report.py`, which writes one markdown file per model holding both
  experiments; `run_all_open_models.sh` calls it as each model finishes.
- Task code (`tasks/`). Only `tasks/layered_vqa/` is defined.
- Model routing (`models.yaml` + `model_registry.py`): which endpoint serves each model.
- Generation settings (`agents.yaml` + `agent_config.py`): what each agent sends —
  temperature, token budget, timeout, retries.
- Web-based viewer to inspect simulated conversations (`app_conv_viewer.py`).
- Prompts (`prompts/`) used during simulation.
- No instruction data lives here; `layered_vqa` reads the dataset selected by
  `LAYERED_VQA_DATASET` (default `../../data/derived/lic/layered_vqa_text.json`),
  exported by `src/vecgen/export_lic_dataset.py` in the parent repository.

## 1. Simulating Conversations

You can simply run the following command:
```sh
python run_simulations.py
```

Which will simulate conversations with all the default parameters (run with `--help` to find out what they are).

### Model routing

Every model is reached through the OpenAI chat-completions API — open-weight models through a local [vLLM server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/), closed-weight models through [OpenRouter](https://openrouter.ai/docs/quickstart), which exposes the same API. `models.yaml` maps each name onto a backend and the id sent on the wire:

```yaml
backends:
  vllm:
    base_url_env: VLLM_BASE_URL
  openrouter:
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY

models:
  local-vlm:         {backend: vllm,       model: local-vlm}
  claude-sonnet-4.5: {backend: openrouter, model: anthropic/claude-sonnet-4.5}
```

The keys under `models:` are what you pass to `--models`, `--judge_model` and `--user_model`, and what is written into the logs — so an upstream id rename does not invalidate old runs. Add a model by editing the file, or pass `backend:id` inline for a one-off:

```sh
python run_simulations.py --models local-vlm --judge_model openrouter:openai/gpt-4o-mini
```

An unrecognised model name is an error rather than a guess, and each model announces its route once at startup:

```
[models] local-vlm -> vllm:local-vlm
[models] gpt-4o-mini -> openrouter:openai/gpt-4o-mini
```

Set `OPENROUTER_API_KEY` to use OpenRouter; clients are built lazily per backend, so a purely local run never needs one. Point the routing table elsewhere with `LAYERED_VQA_MODELS_CONFIG`.

> [!WARNING]
> Any model reached through OpenRouter costs real money. Nothing here tracks spend — check the OpenRouter dashboard.

### Generation settings

`models.yaml` says *where* a call goes; `agents.yaml` says *how* it is made. Three agents make every model call — the **assistant** under test, the simulated **user** that reveals shards, and the **judge** that grades the finished conversation — and each has its own block:

```yaml
agents:
  assistant:
    temperature: 1.0
    top_p: 1.0
    top_k: null                   # omitted from the request; see below
    max_tokens: 1024
    reasoning_max_tokens: 16384   # completion cap for configured reasoning models
    timeout: 600
    max_retries: 3
  user:
    temperature: 1.0
    top_p: 1.0
    top_k: null
    max_tokens: 1024
    reasoning_max_tokens: 16384
    timeout: 300
  judge:
    temperature: 1.0
    top_p: 1.0
    top_k: null
    max_tokens: 1024
    reasoning_max_tokens: 16384
    timeout: 600
```

`top_p` and `top_k` are sent only when they hold a value; `null` leaves the
parameter off the request so the provider's own default applies. `top_k` is not
part of the OpenAI chat-completions schema, so it travels in `extra_body`, which
vLLM and OpenRouter both read from the top level of the body. It ships as `0`,
which both backends read as *disabled* — nothing is truncated, so the shipped
profile samples exactly as it did before the knob existed. Set a positive cap
such as `50` to actually narrow sampling. (vLLM still accepts its older `-1`
spelling too; vLLM before ~0.8.5 predates the change and rejects `0` with a
400.)

All roles and all simulators use one sampling and token-budget profile. The
reasoning cap is only an upper bound: disabling thinking does not require or
produce a reasoning trace; the model returns its direct visible answer. The
settings in force are announced once per agent at startup, next to the routing
lines:

```
[agents] assistant/sharded -> temperature=1.0 top_p=1.0 top_k=None max_tokens=1024 timeout=600s retries=3
[agents] judge -> temperature=1.0 top_p=1.0 top_k=None max_tokens=1024 timeout=600s retries=3
```

An unknown key is an error rather than an ignored line — a misspelled `max_tokens` that "worked" would silently run a whole experiment on the provider's default budget. Point the file elsewhere with `LAYERED_VQA_AGENTS_CONFIG`. `--assistant_temperature` / `--user_temperature` still override the file for one launch.

> [!IMPORTANT]
> Temperature is part of a run's identity. Any assistant or user temperature other than 1.0 adds a suffix to the conv_type a run logs under (`full-t0.7`, `sharded-at0.7-ut1.0`, `snowball-t0.7-scripted`, …), so runs at different temperatures never mix — and changing a temperature starts a fresh set of runs rather than resuming the existing ones. `top_p` and `top_k` get no such suffix: changing either alters how the assistant samples without changing where the run logs, so archive the old logs before changing them mid-experiment.

### Using a local model with vLLM

The simulator can also use a local [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/). Start a chat-capable instruct model, for example:

```sh
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --served-model-name local-model \
  --generation-config vllm
```

To serve Qwen3.5 without internal thinking, configure that mode explicitly on
vLLM (Qwen3.5 does not support the older `/nothink` soft switch):

```sh
vllm serve Qwen/Qwen3.5-9B \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

The answer-only task prompt controls visible response format; it is not the
thinking-mode switch. Evaluation does not require assistant reasoning: the
judge compares visible assistant text and supplies its own short `reasoning`.

Point the client at the server, and make sure `models.yaml` has an entry whose `model:` matches the `--served-model-name` you used:

```sh
export VLLM_BASE_URL=http://localhost:8000/v1

python run_simulations.py \
  --models local-model \
  --judge_model local-model \
  --user_model local-model \
  --tasks layered_vqa \
  --N_workers 1
```

If vLLM was started with `--api-key`, also set the matching key:

```sh
export VLLM_API_KEY=your-vllm-key
```

`VLLM_BASE_URL` overrides the `base_url` in `models.yaml`. The model must provide a chat template and should support JSON-formatted responses because the simulated user agent and the answer judge both request JSON output.

### Using images with a local vLLM model

Serve a vision-language model instead of a text-only instruct model. For
multiple images in one turn, configure the server's multimodal prompt limit to
match the largest turn in the dataset:

```sh
vllm serve <vision-model> \
  --served-model-name local-vlm \
  --limit-mm-per-prompt.image 8

export VLLM_BASE_URL=http://localhost:8000/v1
```

Samples can declare images shared by the whole task and images revealed with a
specific shard. Local paths are resolved relative to the dataset JSON file and
encoded as data URLs immediately before inference, while logs retain the
readable paths. HTTP(S) and pre-encoded data URLs are also accepted.

```json
{
  "task_id": "visual-example-1",
  "task": "layered_vqa",
  "images": ["images/scene.jpg"],
  "shards": [
    {"shard_id": 1, "shard": "What object is visible?"},
    {
      "shard_id": 2,
      "shard": "Focus on the object behind the chair.",
      "images": [{"path": "images/crop.png", "detail": "high"}]
    }
  ]
}
```

In sharded runs, top-level images are sent on the first user turn and shard
images are sent when that shard is revealed. Full and concat runs send all
images in the single user turn. Text-only datasets behave as before.

The model adapter can also be used directly:

```python
from model_openai import generate

response = generate(
    [{"role": "user", "content": "What is shown in this image?"}],
    model="local-vlm",
    images=["images/scene.jpg"],
)
```

## 2. Viewing Simulated Conversations

Use the streamlit app to view conversations:
```sh
streamlit run app_conv_viewer.py
```

## 3. Adding a New Task

You can look at the `task_base.py` file for a sense of the required functions.
Then create a folder in `tasks/<new_task>` with your task logic files, and add an entry into `tasks.py` with the new task class. That's it!

## Dataset Contents

`layered_vqa` samples follow the sharded schema:

```
{
    "task_id": "sharded-{original-task-id}",
    "task": "layered_vqa",
    "shards": [{"shard_id": 1, "shard": "[...]"}, {"shard_id": 2, "shard": "[...]"}, ...],
    "<task_specific_keys>": <task_specific_values>
}
```

The task-specific keys carry the reference answer and the per-sample metadata
(`scene_id`, `category`, `gold_layer_ids`, `distractor_layer_ids`,
`candidate_layer_ids`, `demand_level`) used for bookkeeping and evaluation.

`meta.response_format` decides what the assistant is asked for. The default,
`"answer"`, is the short-phrase answer every condition has always wanted. The
`grounding` export sets `"answer_and_layers"`, which serves the layer-selection
prompt — answer plus the supporting layer ids — and makes the task score the
predicted evidence set alongside the answer verdict.

