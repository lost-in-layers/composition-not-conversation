# Layered-VQA Shard Review

Local single-reviewer UI for reviewing generated question shards and
multimodal layer-evidence annotations for all 300 accepted questions. It reads
the latest valid checkpoint record for every question.

```powershell
powershell -ExecutionPolicy Bypass -File vendor\shard-review\run.ps1 -Port 8793
```

The UI provides the same full-resolution viewer, fit/1:1 controls, layer rail,
alpha view, overlay view, progressive strip, and geometry overlay as the
category reviewer. Below it, every generated field is visible and editable.
Draft edits autosave; accepting an entry enforces the 3–6 shard and
layer-evidence consistency checks. **Next flagged** opens model-flagged records
first.

Drafts and decisions autosave under `vendor/shard-review/runtime/`. Accepted
reviewed questions are atomically exported to:

```text
data/derived/mulan-human-review/reviewer_2/question-shards-reviewed.json
```

The source `accepted_scenes.json` and `question_shards.jsonl` are never changed.
