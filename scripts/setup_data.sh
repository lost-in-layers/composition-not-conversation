#!/usr/bin/env bash
# Put the data that vendor/conv-smi/run_all_open_models.sh needs onto a fresh
# machine.
#
#   bash scripts/setup_data.sh            # ~110 MB, exactly what the runner reads
#   FULL=1 bash scripts/setup_data.sh     # ~2.2 GB, every scene in the pool
#
# Run it from anywhere; it finds the repository from its own path. Re-running is
# safe and is the fix for an interrupted download -- the client resumes.
#
# Why the default is small. data/selected_1200 holds 1,200 scenes, but the eight
# task files the runner loads only ever reference 93 of them, and only their
# full.png. Everything else in that tree -- the per-scene layer PNGs, the
# progressive strips, the other 1,107 scenes -- belongs to dataset generation,
# not to the experiment. Fetching the whole pool costs 2.1 GB the runner never
# opens, so the include list is computed from the task JSONs instead of guessed.
# Use FULL=1 if you intend to regenerate or extend the datasets on this machine.
#
# Destinations are load-bearing: image paths inside data/derived/lic/*.json are
# relative to that JSON's own directory, so a moved directory breaks every
# conversation that shows an image.
#
#   HF data/derived/lic_renders/ -> data/derived/lic_renders/   (93 scenes, layer PNGs)
#   HF laion/<id>/, lvis/<id>/   -> data/selected_1200/...      (the recap full.png)
#
# The task JSONs themselves are tracked in git and arrive with the clone. They
# are not downloaded here, only verified.

set -euo pipefail

REPO_ID="${REPO_ID:-lost-in-layer/Layered-VQA}"
FULL="${FULL:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "Repository root: $ROOT"

# huggingface_hub 1.x ships `hf` and has dropped `huggingface-cli`; older
# installs only have the latter. Accept either.
if command -v hf >/dev/null 2>&1; then
  HF=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF=(huggingface-cli download)
else
  echo "Neither 'hf' nor 'huggingface-cli' is on PATH." >&2
  echo "Install it with:  pip install -U 'huggingface_hub[cli]'" >&2
  exit 1
fi
echo "Downloader: ${HF[*]}"

if [[ ! -d data/derived/lic ]]; then
  echo "data/derived/lic is missing -- this is not a complete clone of the repository." >&2
  exit 1
fi

# The data is hosted as a Hugging Face model repo, not a dataset repo, so the
# type is pinned rather than left to the client default.
fetch() {
  local dest="$1"; shift
  mkdir -p "$dest"
  "${HF[@]}" "$REPO_ID" --repo-type model --local-dir "$dest" --include "$@"
}

echo
echo "=== 1/2  layer renders -> data/derived/lic_renders/ (~56 MB) ==="
# These HF paths already begin with data/, so the destination is the repo root
# and they land in their final location directly.
fetch . 'data/derived/lic_renders/*'

echo
echo "=== 2/2  scene images -> data/selected_1200/ ==="
if [[ "$FULL" == "1" ]]; then
  echo "FULL=1: fetching the entire scene pool (~2.2 GB)"
  fetch data/selected_1200 'laion/*' 'lvis/*' 'manifest.csv'
else
  # Read the include list out of the task files, so it cannot drift from what
  # the runner will actually open.
  mapfile -t PATTERNS < <(python - <<'PY'
import json
from pathlib import Path

lic = Path("data/derived/lic")
wanted = set()
for task in sorted(lic.glob("layered_vqa_*.json")):
    if task.name.endswith(".manifest.json"):
        continue
    data = json.loads(task.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else list(data.values())
    for row in rows:
        buckets = [row.get("images") or [], row.get("recap_images") or []]
        buckets += [s.get("images") or [] for s in (row.get("shards") or [])]
        for bucket in buckets:
            for img in bucket:
                # Repo-relative, forward slashes, as an HF include pattern.
                p = (lic / img["path"]).resolve().relative_to(Path.cwd()).as_posix()
                if p.startswith("data/selected_1200/"):
                    wanted.add(p[len("data/selected_1200/"):])
for p in sorted(wanted):
    print(p)
PY
)
  echo "${#PATTERNS[@]} scene images referenced by the task files"
  fetch data/selected_1200 "${PATTERNS[@]}"
fi

# --- verification -----------------------------------------------------------
# Counting directories proves nothing: an interrupted download leaves scene
# folders holding some of their layers. What matters is whether every image path
# the runner's datasets reference resolves on disk, so that is what is checked.
echo
echo "=== verifying ==="
python - <<'PY'
import json, sys
from pathlib import Path

lic = Path("data/derived/lic")

# The eight datasets run_all_open_models.sh loads, plus the shuffled-text file
# used by SHUFFLED_CONCAT.
RUNTIME = [
    "layered_vqa_text.json", "layered_vqa_text_shuffled.json",
    "layered_vqa_visual_canonical.json", "layered_vqa_grounding.json",
    "layered_vqa_oracle_gold.json", "layered_vqa_aligned_vq.json",
    "layered_vqa_misaligned_vq.json",
    "layered_vqa_visual_evidence_first.json", "layered_vqa_visual_evidence_last.json",
]

missing_tasks = [n for n in RUNTIME if not (lic / n).is_file()]
for n in missing_tasks:
    print(f"  [FAIL] task file missing: {n}")

refs = 0
broken = []
for name in RUNTIME:
    task = lic / name
    if not task.is_file():
        continue
    data = json.loads(task.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else list(data.values())
    for row in rows:
        buckets = [row.get("images") or [], row.get("recap_images") or []]
        buckets += [s.get("images") or [] for s in (row.get("shards") or [])]
        for bucket in buckets:
            for img in bucket:
                refs += 1
                if not (lic / img["path"]).resolve().is_file():
                    broken.append(f"{name}: {img['path']}")

print(f"  [{'ok' if not broken else 'FAIL'}] image references   {refs:,} checked, "
      f"{'all resolve' if not broken else f'{len(broken):,} BROKEN'}")
for b in broken[:5]:
    print(f"        {b}")
if len(broken) > 5:
    print(f"        ... and {len(broken) - 5:,} more")

ok = not broken and not missing_tasks
print()
print("Data is ready -- run_all_open_models.sh has everything it reads."
      if ok else "Setup incomplete -- re-run this script; downloads resume.")
sys.exit(0 if ok else 1)
PY
