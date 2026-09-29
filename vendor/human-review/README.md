# MuLAn Human-Review Tool (local only)

A localhost-only application for **two human reviewers** to curate the curated
**`data/selected_1200`** MuLAn scene decompositions for the Layered-VQA pilot:
verify layers, complete the intentionally-null canonical labels, fix captions,
check geometry/relations, assign question categories, and accept or reject
scenes.

The 1200 selected scenes are split **deterministically and disjointly** between
two fixed reviewer slots (Reviewer 1 / Reviewer 2), so each reviewer owns a
fixed, non-overlapping set of images with its own queue, state and exports.
Together the two reviewers cover the whole selection exactly once.

Everything runs on `127.0.0.1`, reads/writes ordinary files inside this
repository, and never talks to any external service. There is no database, no
Docker, no deployment configuration, and no question generation.

---

## Requirements

- Windows PowerShell
- Python 3.11+ on `PATH` (`python --version`)
- Node.js 18+ with npm (`node --version`)

## Setup (one time)

```powershell
powershell -ExecutionPolicy Bypass -File vendor\human-review\setup.ps1
```

This creates `vendor/human-review/.venv`, installs the backend requirements
(FastAPI, uvicorn, ijson, pytest), runs `npm install`, and builds the static
frontend into `frontend/dist/`.

## Launch

```powershell
powershell -ExecutionPolicy Bypass -File vendor\human-review\run.ps1
```

The backend binds to `http://127.0.0.1:8791` (change with `-Port 9000`) and the
default browser opens automatically. Stop with `Ctrl+C`.

## First run

1. **Build the index.** The app asks you to build the local scene index. It
   streams both source files once with `ijson` (they are ~270 MB each and are
   never loaded whole), keeps **only** the scenes present under
   `data/selected_1200/`, repoints their render paths to that curated copy, and
   writes a seekable cache. Takes a minute or two.
2. **Pick a reviewer.** Choose **Reviewer 1** or **Reviewer 2**. Each slot has a
   fixed, non-overlapping ~600-image assignment and its own state and exports.
   The choice is remembered across restarts; switch any time from the header.
3. **The Review Guide opens.** Read it once; tick *Do not automatically show
   again* if you wish. Reopen it any time with the header button, the `g` key,
   or the `?` help icons next to each step title (those deep-link to the
   relevant section).

---

## Files

### Source (read-only — never modified)

```
data/derived/mulan-scenes/lvis_scenes.json    scene metadata for lvis_* scenes
data/derived/mulan-scenes/laion_scenes.json   scene metadata for laion_* scenes
data/selected_1200/<subset>/<image_id>/full.png, layer_*.png, progressive_strip.png
data/selected_1200/manifest.csv               (informational; the directories are authoritative)
```

Two independent inputs are joined:

- **`data/selected_1200/`** defines *which* scenes are reviewable (only scenes
  whose renders are present here are indexed/queued) and *where the images are*
  (each scene's render paths are repointed to this curated copy). Its
  `<image_id>` directory name equals the scene records' `image_id`, so the scene
  id is `f"{subset}_{image_id}"`.
- **`data/derived/mulan-scenes/*.json`** supplies every non-image field —
  captions, labels, geometry, relations, categories. `selected_1200` carries no
  such metadata, so these JSONs must be present; build them with
  `python -m src.vecgen.build_mulan_scene_records` if missing.

A selected scene that is missing from the scene JSONs is reported as a warning
and skipped (it cannot be reviewed until its record is added). The original
`data/pre-data/renders/` tree is **not** used by the review tool.

### Runtime (local, gitignored, safe to regenerate)

```
vendor/human-review/runtime/cache/lvis_scenes.jsonl    one compact (repointed) scene per line
vendor/human-review/runtime/cache/laion_scenes.jsonl
vendor/human-review/runtime/cache/scene_index.json     byte offset + length per scene
vendor/human-review/runtime/review_queue_r1.json       Reviewer 1's fixed deterministic queue
vendor/human-review/runtime/review_queue_r2.json       Reviewer 2's fixed deterministic queue
vendor/human-review/runtime/review_state_r1.json       Reviewer 1: cursor, drafts, decisions, undo
vendor/human-review/runtime/review_state_r2.json       Reviewer 2: cursor, drafts, decisions, undo
vendor/human-review/runtime/active_reviewer.json       which slot is currently selected
vendor/human-review/runtime/backups/                   timestamped state backups
```

Scenes are fetched by seeking directly into the JSONL cache — nothing keeps
all scenes in memory, and the browser only ever receives one scene at a time.

### Output (the deliverables — one folder per reviewer)

```
data/derived/mulan-human-review/reviewer_1/accepted_scenes.json
data/derived/mulan-human-review/reviewer_1/rejected_scenes.json
data/derived/mulan-human-review/reviewer_2/accepted_scenes.json
data/derived/mulan-human-review/reviewer_2/rejected_scenes.json
```

Each reviewer's pair is rewritten **atomically after every final decision**
(temp file + rename), so they are always valid JSON and always current. Every
record carries the `reviewer` id (`reviewer_1` / `reviewer_2`). `Export now` in
the header forces a rewrite of the active reviewer's exports on demand.

---

## Review workflow

The screen is split: the upper ~57% always shows the visuals (full image,
background, layers, alpha masks, overlay/progressive-strip modes, zoom/pan/fit,
geometry overlay); the lower part is a five-step workflow:

1. **Layer Quality** — rate the decomposition 1–5; mark every foreground layer
   *Keep* or *Exclude* (with a reason). The
   background cannot be excluded — reject the scene instead. Immediate
   rejection is available here.
2. **Captions & Labels** — edit the existing BLIP-2/LLaVA overall and
   background captions directly, and supply a canonical short label
   (`giraffe`, `red bag`…) for every retained layer. Each layer's BLIP-2
   caption is directly editable. `label_source` becomes `human_review` on
   acceptance.
3. **Geometry** — verify bboxes/centroids/z-order; drag boxes, corners, or the
   centroid in the viewer (or type exact values); mark each retained layer
   correct / minor correction / major problem; mark relations invalid or add
   new ones. Corrections are stored as **overrides** — original geometry and
   masks are untouched.
4. **Categories** — tick every suitable category, choose exactly one primary.
   Live per-category progress toward 100 is shown; a full quota warns but the
   human decision stays authoritative. All six categories remain selectable;
   automatic suggestions never block the reviewer.
5. **Final Decision** — Step 5 automatically shows one question-and-answer area
   for every category selected in Step 4. Complete at least one pair; every
   question you fill must also have an answer. Review the summary, then
   *Accept & Next*. Rejection happens directly in Step 1 and opens the next
   scene after confirmation.

Autosave runs ~1 s after every change (status is visible in the header).
Navigation, restarts and jumps never lose work.

### Null / missing-field handling

Every scene is classified on load:

- **Must be completed by you** (`n missing` badges): canonical labels
  (`label: null` in the source), reviewed captions, layer decisions, geometry
  status, primary category.
- **Legitimately null technical fields** (`tech-null` badge): e.g.
  `mask.native_alpha_sha256` / `matches_native_alpha` when no native checksum
  existed, `annotation.actual_sha256`, empty `inventory.error`, empty LAION
  `categories`. These are preserved verbatim in provenance — never invent
  values for them.
- **Blocking source gaps** (red badges): missing image files or layers with no
  source caption at all. Resolve by writing the missing caption yourself,
  excluding the layer, rejecting, or deferring.

Accepted scenes are validated twice (at acceptance and again at export):
no null/empty required field can reach `accepted_scenes.json`.

### Queue (per reviewer, fixed disjoint split)

Seed **20260719** (`queueing.DEFAULT_SEED`; only affects a *new* queue). Within
each subset the selected scene ids are shuffled with the seed and split even/odd
— Reviewer 1 takes indices 0, 2, 4… and Reviewer 2 takes 1, 3, 5…; each
reviewer's LVIS and LAION halves are then interleaved 1:1 (LVIS first). The
split is a pure function of the selection + seed, so a given scene always lands
with the same reviewer, and the two reviewers' sets are disjoint and together
cover the whole selection once (roughly 600 each for 1200 selected).

Each reviewer's order is written once to `runtime/review_queue_<slot>.json` and
**never regenerated on restart** — deleting that file is the only way to get a
new order, and finalized scenes are skipped wherever the cursor lands. Jumping
to a scene ID never moves the queue cursor; *Resume* returns to it.

### Keyboard shortcuts (press `?` in the app)

`g` guide · `1–5` steps · `[` / `]` prev/next scene · `d` defer · `Esc` close
dialogs.

---

## Progress & dashboard

The header shows the active reviewer, **reviewed / assigned** (finalized out of
that reviewer's ~600), accepted/rejected/deferred/drafts, LVIS/LAION balance and
autosave status. Use **Switch** in the header to change reviewer. The
**Dashboard** button adds per-category progress bars, quality distribution,
rejection/exclusion reason histograms, caption edit rate, missing-label
completion rate, geometry correction rate and mean review duration — all for the
active reviewer only.

## Backups, undo, safety

- **Backup** writes `runtime/backups/review_state_<slot>.<UTC-timestamp>.json`
  for the active reviewer.
- **Undo last** reverts the active reviewer's most recent final decision (scene
  returns to draft; that reviewer's exports are rewritten).
- All state files are written atomically (temp file + `os.replace`), so a
  crash can never corrupt them.
- The image endpoint only serves `data/**` image files inside the repository;
  path traversal is rejected.

## Rebuilding the cache (safe)

If the source JSONs are regenerated the header shows a warning. Rebuild via:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8791/api/index/rebuild
```

(or delete `runtime/cache/` and restart — the app offers the build screen).
Review decisions, drafts, the per-reviewer queues and exports are **never**
touched by a cache rebuild. The selection is re-read from `data/selected_1200`
on every rebuild, but each reviewer's queue order is fixed once written; newly
added scenes only enter a *new* queue (delete `review_queue_<slot>.json`).

## Intentional full reset

Only if you really want to discard all review work:

1. Stop the app.
2. Delete `vendor/human-review/runtime/` (optionally keep `backups/`).
3. Delete `data/derived/mulan-human-review/` if the exports should go too.
4. Restart — the app rebuilds the cache and a fresh queue.

## Tests

```powershell
# backend (60 tests — uses small fixtures, never the real 270 MB files)
vendor\human-review\.venv\Scripts\python.exe -m pytest vendor\human-review\backend\tests -q

# frontend (32 tests)
cd vendor\human-review\frontend; npm test
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Backend unreachable` in the browser | Start `run.ps1`; check the port is free (`netstat -ano \| findstr 8791`). |
| `Frontend not built` (503) | Run `setup.ps1` (or `npm run build` in `frontend/`). |
| `Source file missing` on the index screen | The app expects `data/derived/mulan-scenes/*.json` and `data/selected_1200/` relative to the repo root. Set `HR_REPO_ROOT` if you run the backend from an unusual location. |
| `No selected scenes found` on rebuild | `data/selected_1200/<subset>/<image_id>/full.png` is missing — restore the curated selection directory. |
| Warning: N selected scenes missing from the scene JSON | Those `selected_1200` scenes have no metadata record. Add them to `data/derived/mulan-scenes/` and rebuild; until then they are skipped. |
| Warning: source changed since cache was built | Rebuild the index (see above). Decisions are preserved. |
| Images 404 / red "missing file" badges | The scene's folder under `data/selected_1200/` is incomplete — exclude/reject/defer; this usually means files were moved. |
| Port already in use | `run.ps1 -Port 9000`. |
| Reset the guide auto-show | Untick *Do not automatically show again* in the guide footer. |

## Architecture (for maintainers)

- `backend/hr_backend/paths.py` — repo-root discovery, per-reviewer file paths, safe image-path resolution.
- `atomicio.py` — atomic JSON writes.
- `selection.py` — reads `data/selected_1200` (which scenes are reviewable) and repoints scene image paths to that curated copy.
- `reviewers.py` — the two fixed reviewer slots (`r1`/`r2`), ids and labels.
- `indexer.py` — ijson streaming → JSONL cache + offset index; filters to the selection and repoints images; freshness/selection warnings; seek-based fetch.
- `missing.py` — null/empty classification (human-required vs technical vs blocking).
- `queueing.py` — seeded shuffle + even/odd split + interleave into a fixed disjoint per-reviewer queue; cursor navigation.
- `state.py` — per-reviewer `review_state_<slot>.json` store (drafts, deferrals, decisions, undo, backups).
- `curation.py` — acceptance/rejection validation, advisory category suggestions, curated-scene builder, export-time validation.
- `exports.py` — atomic per-reviewer export rewriting.
- `progress.py` — reviewed/assigned + dashboard statistics.
- `service.py` — orchestration incl. reviewer selection; `api.py` — FastAPI routes; `main.py` — uvicorn on 127.0.0.1.
- `frontend/src/` — React/TS app: `VisualPane` (viewer + layer rail + geometry overlay), five step components, `GuideModal`, `Dashboard`, `draftUtils` (client-side mirrors of the validation rules).
