# MuLAn Bootstrap and Storage-Safe Rebuild (Ubuntu Server)

This runbook starts from the public MuLAn multipart archive, rebuilds enriched
lossless inventories, links the pre-rendered layer archives, verifies retained
data, and removes only disposable download archives.

Run these commands on the Ubuntu server, not on a laptop.

## What must be retained

Keep:

- extracted `.p.zl` files: the canonical lossless MuLAn annotations
- enriched inventory CSVs
- extracted `full.png` and `layer_*.png` renders
- `file_id_url_linking.csv`
- COCO annotation JSON if image-level categories or mask matching are needed

Delete only after the corresponding verification succeeds:

- `mulan.part001.rar` through the final multipart RAR
- `annotations_trainval2017.zip`
- `lvis.zip` and `laion.zip` after extraction

The inventory does not duplicate native mask/delta arrays into CSV cells. That
would duplicate much of the annotation corpus and make CSV processing fragile.
Instead, it copies every scalar/text field, records a recursive manifest with
shape/dtype/size/SHA-256 for each binary value, and stores the SHA-256 and path
of the original `.p.zl`. Together with the retained `.p.zl`, this is lossless
and verifiable.

## 1. Set server paths and install prerequisites

The commands below use the current server checkout at `/root/Layered-VQA` and
store generated/downloaded data under its ignored `data/pre-data` directory.

```bash
set -euo pipefail

export REPO_ROOT=/root/Layered-VQA
export DATA_ROOT=/root/Layered-VQA/data/pre-data
export DOWNLOAD_ROOT="$DATA_ROOT/downloads"
export MULAN_DOWNLOAD="$DOWNLOAD_ROOT/mulan"
export RENDER_DOWNLOAD="$DOWNLOAD_ROOT/renders"
export ANNOTATION_ROOT="$DATA_ROOT/mulan_v1/extracted"
export COCO_ANNOTATION_ROOT="$DATA_ROOT/coco_annotations"
export RENDER_ROOT="$DATA_ROOT/renders"

cd "$REPO_ROOT"
mkdir -p \
  "$MULAN_DOWNLOAD" \
  "$RENDER_DOWNLOAD" \
  "$ANNOTATION_ROOT" \
  "$COCO_ANNOTATION_ROOT" \
  "$RENDER_ROOT"

apt-get update
apt-get install -y python3-venv unrar unzip curl

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-mulan.txt

command -v hf
command -v unrar
command -v unzip
```

## 2. Download MuLAn annotations and URL map

The multipart download is large. `hf download` resumes partial downloads, so
rerun the same command after a network interruption.

```bash
hf download mulan-dataset/v1.0 \
  --repo-type dataset \
  --include 'mulan.part*.rar' \
  --include 'file_id_url_linking.csv' \
  --local-dir "$MULAN_DOWNLOAD"

export FIRST_PART="$MULAN_DOWNLOAD/mulan.part001.rar"
test -f "$FIRST_PART"

RAR_COUNT=$(find "$MULAN_DOWNLOAD" -maxdepth 1 -type f \
  -name 'mulan.part*.rar' -printf '.' | wc -c)
if [ "$RAR_COUNT" -ne 350 ]; then
  echo "Expected 350 MuLAn RAR parts, found $RAR_COUNT." >&2
  exit 1
fi

unrar t "$FIRST_PART"
```

`unrar t` must finish successfully before extraction.

## 3. Extract and verify native `.p.zl` annotations

Open only `part001`; `unrar` automatically reads the remaining parts. Use `x`
so the subset directory structure is preserved.

```bash
unrar x -o+ "$FIRST_PART" "$ANNOTATION_ROOT/"

export PZL_COUNT
PZL_COUNT=$(find "$ANNOTATION_ROOT" -type f -name '*.p.zl' -printf '.' | wc -c)
if [ "$PZL_COUNT" -ne 44860 ]; then
  echo "Expected 44,860 .p.zl files, found $PZL_COUNT. Do not delete the RAR files." >&2
  exit 1
fi

export LVIS_PZL
export LAION_PZL
LVIS_PZL=$(find "$ANNOTATION_ROOT" -type d -name lvis -print -quit)
LAION_PZL=$(find "$ANNOTATION_ROOT" -type d -name laion_aesthetic_v2_6.5 -print -quit)
test -n "$LVIS_PZL" && test -d "$LVIS_PZL"
test -n "$LAION_PZL" && test -d "$LAION_PZL"

mkdir -p "$DATA_ROOT/mulan_v1"
cp -- "$MULAN_DOWNLOAD/file_id_url_linking.csv" \
  "$DATA_ROOT/mulan_v1/file_id_url_linking.csv"
```

## 4. Download the small COCO annotation package

Source images are not needed when using the pre-rendered layer ZIPs. COCO
annotation JSON is used to populate image-level candidate categories and,
later, match COCO instance masks to MuLAn layers.

```bash
export COCO_ZIP="$DOWNLOAD_ROOT/annotations_trainval2017.zip"
curl -fL --retry 5 --retry-delay 5 \
  'https://images.cocodataset.org/annotations/annotations_trainval2017.zip' \
  -o "$COCO_ZIP"

unzip -t "$COCO_ZIP"
unzip -o "$COCO_ZIP" -d "$COCO_ANNOTATION_ROOT"

export COCO_JSON_DIR="$COCO_ANNOTATION_ROOT/annotations"
test -f "$COCO_JSON_DIR/instances_train2017.json"
test -f "$COCO_JSON_DIR/instances_val2017.json"

rm -- "$COCO_ZIP"
```

## 5. Build enriched inventories

These commands export native captions and the recursive annotation manifest.
They also calculate and store every `.p.zl` SHA-256.

```bash
export INVENTORY_ROOT="$DATA_ROOT/inventories"
export STATS_ROOT="$DATA_ROOT/stats"
export CATEGORY_CACHE="$DATA_ROOT/mulan_v1/coco_image_categories.json"
export URL_MAP="$DATA_ROOT/mulan_v1/file_id_url_linking.csv"
mkdir -p "$INVENTORY_ROOT" "$STATS_ROOT"

python -m src.vecgen.mulan_dataset_tool inventory \
  --subset lvis \
  --pzl-dir "$LVIS_PZL" \
  --inventory "$INVENTORY_ROOT/lvis.csv" \
  --category-cache "$CATEGORY_CACHE" \
  --coco-annotations-dir "$COCO_JSON_DIR" \
  --workers 12

python -m src.vecgen.mulan_dataset_tool inventory \
  --subset laion_aesthetic_v2_6.5 \
  --pzl-dir "$LAION_PZL" \
  --inventory "$INVENTORY_ROOT/laion.csv" \
  --url-map "$URL_MAP" \
  --workers 12

python -m src.vecgen.mulan_dataset_tool stats \
  --inventory "$INVENTORY_ROOT/lvis.csv" \
  --output-dir "$STATS_ROOT/lvis"

python -m src.vecgen.mulan_dataset_tool stats \
  --inventory "$INVENTORY_ROOT/laion.csv" \
  --output-dir "$STATS_ROOT/laion"
```

If the server has fewer CPU cores or limited RAM, reduce `--workers 12`.

## 6. Verify inventories, then delete multipart RARs

Do not run the deletion block unless both verification commands finish with
`Verified` and exit code 0.

```bash
python -m src.vecgen.mulan_dataset_tool verify \
  --inventory "$INVENTORY_ROOT/lvis.csv" \
  --workers 12

python -m src.vecgen.mulan_dataset_tool verify \
  --inventory "$INVENTORY_ROOT/laion.csv" \
  --workers 12
```

Guard the deletion target, recount the archive set, and then delete only the
multipart RAR files:

```bash
resolved_data_root=$(realpath "$DATA_ROOT")
resolved_mulan_download=$(realpath "$MULAN_DOWNLOAD")
case "$resolved_mulan_download" in
  "$resolved_data_root"/*) ;;
  *) echo 'Refusing deletion outside DATA_ROOT.' >&2; exit 1 ;;
esac

rar_count=$(find "$resolved_mulan_download" -maxdepth 1 -type f \
  -name 'mulan.part*.rar' -printf '.' | wc -c)
if [ "$rar_count" -ne 350 ]; then
  echo "Refusing to delete an unexpected RAR set of $rar_count files." >&2
  exit 1
fi

find "$resolved_mulan_download" -maxdepth 1 -type f \
  -name 'mulan.part*.rar' -print -delete
```

This does not delete the extracted `.p.zl` annotations.

## 7. Download, extract, and delete render ZIPs one at a time

Processing one ZIP at a time avoids storing both large compressed render
archives alongside their extracted contents.

### LVIS/COCO renders

```bash
hf download lost-in-layer/Layered-VQA 'renders/lvis.zip' \
  --local-dir "$RENDER_DOWNLOAD"
export LVIS_ZIP="$RENDER_DOWNLOAD/renders/lvis.zip"

unzip -t "$LVIS_ZIP"
mkdir -p "$RENDER_ROOT"
# The ZIP already contains its leading lvis/ directory.
unzip -o "$LVIS_ZIP" -d "$RENDER_ROOT"
export LVIS_RENDER_ROOT="$RENDER_ROOT/lvis"

LVIS_FULL_COUNT=$(find "$LVIS_RENDER_ROOT" -type f -name full.png \
  -printf '.' | wc -c)
if [ "$LVIS_FULL_COUNT" -eq 0 ]; then
  echo 'No LVIS full.png files were extracted.' >&2
  exit 1
fi

rm -- "$LVIS_ZIP"
```

### LAION renders

```bash
hf download lost-in-layer/Layered-VQA 'renders/laion.zip' \
  --local-dir "$RENDER_DOWNLOAD"
export LAION_ZIP="$RENDER_DOWNLOAD/renders/laion.zip"

unzip -t "$LAION_ZIP"
mkdir -p "$RENDER_ROOT"
# The ZIP already contains its leading laion/ directory.
unzip -o "$LAION_ZIP" -d "$RENDER_ROOT"
export LAION_RENDER_ROOT="$RENDER_ROOT/laion"

LAION_FULL_COUNT=$(find "$LAION_RENDER_ROOT" -type f -name full.png \
  -printf '.' | wc -c)
if [ "$LAION_FULL_COUNT" -eq 0 ]; then
  echo 'No LAION full.png files were extracted.' >&2
  exit 1
fi

rm -- "$LAION_ZIP"
```

### Repair archives extracted one directory too deep

Skip this section for a fresh extraction using the commands above. Older
commands extracted each ZIP inside its own leading directory and produced
`lvis/lvis/lvis` and `laion/laion/laion_aesthetic_v2_6.5`. The following
same-filesystem directory renames repair that layout without copying or
re-rendering the PNG stacks:

```bash
export LVIS_MIDDLE="$RENDER_ROOT/lvis/lvis"
export LVIS_SOURCE="$LVIS_MIDDLE/lvis"
export LVIS_TEMP="$RENDER_ROOT/lvis/.lvis-relevel-tmp"

export LAION_MIDDLE="$RENDER_ROOT/laion/laion"
export LAION_SOURCE="$LAION_MIDDLE/laion_aesthetic_v2_6.5"
export LAION_TARGET="$RENDER_ROOT/laion/laion_aesthetic_v2_6.5"
export LAION_TEMP="$RENDER_ROOT/laion/.laion-relevel-tmp"

test -d "$LVIS_SOURCE"
test -d "$LAION_SOURCE"
test ! -e "$LVIS_TEMP"
test ! -e "$LAION_TEMP"
test ! -e "$LAION_TARGET"
test "$(find "$LVIS_SOURCE" -type f -name full.png -printf x | wc -c)" \
  -eq 16034
test "$(find "$LAION_SOURCE" -type f -name full.png -printf x | wc -c)" \
  -eq 28826
test "$(find "$LVIS_MIDDLE" -mindepth 1 -maxdepth 1 -printf x | wc -c)" \
  -eq 1
test "$(find "$LAION_MIDDLE" -mindepth 1 -maxdepth 1 -printf x | wc -c)" \
  -eq 1

mv -- "$LVIS_SOURCE" "$LVIS_TEMP"
rmdir -- "$LVIS_MIDDLE"
mv -- "$LVIS_TEMP" "$LVIS_MIDDLE"

mv -- "$LAION_SOURCE" "$LAION_TEMP"
rmdir -- "$LAION_MIDDLE"
mv -- "$LAION_TEMP" "$LAION_TARGET"
```

## 8. Link existing renders into the inventories

`--existing-only` prevents downloading original COCO/LAION images or trying to
regenerate missing stacks.

```bash
python -m src.vecgen.mulan_dataset_tool render \
  --inventory "$INVENTORY_ROOT/lvis.csv" \
  --output-dir "$LVIS_RENDER_ROOT" \
  --existing-only \
  --workers 12

python -m src.vecgen.mulan_dataset_tool render \
  --inventory "$INVENTORY_ROOT/laion.csv" \
  --output-dir "$LAION_RENDER_ROOT" \
  --existing-only \
  --workers 12
```

Verify every row linked successfully:

```bash
python -m src.vecgen.mulan_dataset_tool verify \
  --inventory "$INVENTORY_ROOT/lvis.csv" \
  --render-status existing \
  --require-renders \
  --workers 12

python -m src.vecgen.mulan_dataset_tool verify \
  --inventory "$INVENTORY_ROOT/laion.csv" \
  --render-status existing \
  --require-renders \
  --workers 12
```

Rows marked `missing` still retain their complete native `.p.zl` metadata; they
simply lack a pre-rendered PNG stack in the downloaded archive.

## 9. Final retained-data check

```bash
python - <<'PY'
import csv
import os
from collections import Counter
from pathlib import Path

data_root = Path(os.environ['DATA_ROOT'])
inventory_root = Path(os.environ['INVENTORY_ROOT'])
annotation_root = Path(os.environ['ANNOTATION_ROOT'])

def load_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

lvis_inventory = load_rows(inventory_root / 'lvis.csv')
laion_inventory = load_rows(inventory_root / 'laion.csv')
lvis_rows = len(lvis_inventory)
laion_rows = len(laion_inventory)
lvis_status = Counter(row['render_status'] for row in lvis_inventory)
laion_status = Counter(row['render_status'] for row in laion_inventory)
pzl_count = sum(1 for _ in annotation_root.rglob('*.p.zl'))
lvis_full = sum(
    1 for _ in (data_root / 'renders' / 'lvis' / 'lvis').glob('*/full.png')
)
laion_full = sum(
    1 for _ in (
        data_root / 'renders' / 'laion' / 'laion_aesthetic_v2_6.5'
    ).glob('*/full.png')
)

print({
    'lvis_inventory_rows': lvis_rows,
    'laion_inventory_rows': laion_rows,
    'total_inventory_rows': lvis_rows + laion_rows,
    'pzl_files': pzl_count,
    'lvis_full_renders': lvis_full,
    'laion_full_renders': laion_full,
    'lvis_render_status': lvis_status,
    'laion_render_status': laion_status,
})

if (lvis_rows, laion_rows, pzl_count) != (16034, 28826, 44860):
    raise SystemExit('Final annotation/inventory counts are incorrect.')
if (lvis_full, laion_full) != (16034, 28826):
    raise SystemExit('Final render counts or directory layouts are incorrect.')
if lvis_status != Counter({'existing': 16034}):
    raise SystemExit('LVIS render statuses are incomplete.')
if laion_status != Counter({'existing': 28826}):
    raise SystemExit('LAION render statuses are incomplete.')

print('All 44,860 annotations and render stacks are correctly linked.')
PY
```

At this point the disposable archives are gone. Retain `.p.zl`, inventory CSV,
COCO JSON, URL map, and extracted renders for scene-record generation.
