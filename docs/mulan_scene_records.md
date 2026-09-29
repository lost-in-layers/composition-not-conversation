# MuLAn Scene Records

`src/vecgen/build_mulan_scene_records.py` converts the verified MuLAn
inventories and RGBA render stacks into exactly two readable JSON files:

```text
data/derived/mulan-scenes/lvis_scenes.json
data/derived/mulan-scenes/laion_scenes.json
```

The existing inventory CSVs, `.p.zl` files, `full.png` files, and layer PNGs
are read-only inputs. The builder creates neither inline RLE masks nor separate
mask PNGs. Each instance record references the alpha channel of its existing
RGBA `layer_N.png` and records both its rendered and native alpha SHA-256.

## Build both collections

Run from the repository root after both inventories report every row as
`existing`:

```bash
python -m src.vecgen.build_mulan_scene_records --workers 8
```

To build a training-ready dataset containing only valid scenes, while keeping
excluded IDs and reasons in the same two JSON files, run:

```bash
python -m src.vecgen.build_mulan_scene_records \
  --workers 8 \
  --exclude-invalid \
  --overwrite
```

With `--exclude-invalid`, the main `scenes` array contains only usable records.
The top-level `excluded_scenes` array and `exclusion_reason_counts` summary keep
the filtering decision auditable without creating another artifact.

The defaults read:

```text
data/pre-data/inventories/lvis.csv
data/pre-data/inventories/laion.csv
```

and write both outputs under `data/derived/mulan-scenes/`. Paths stored inside
the JSON are relative to the repository root by default. Use `--overwrite` to
atomically replace existing output files:

```bash
python -m src.vecgen.build_mulan_scene_records \
  --workers 8 \
  --overwrite
```

Use fewer workers if server memory is constrained; some LAION scenes exceed
ten megapixels. Composite validation is enabled by default and verifies that
numeric layer order exactly reconstructs `full.png`.

## Contents

Each file is a UTF-8, two-space-indented JSON object with a `scenes` array and
a final `summary`. Each scene contains:

- native image, background, and per-instance captions;
- captioning-model configuration and the complete inventory metadata manifest;
- checksum-verified `.p.zl`, full-render, render-directory, and numerically
  ordered layer paths;
- background-first ordinal `z_index` values;
- alpha-mask paths, thresholds, dimensions, and checksums;
- half-open bounding boxes, centroids, support/core/soft areas, perimeter,
  connected components, occupancy, and border contact;
- pairwise intersections, IoU, overlap fractions, and centroid distances;
- conservative spatial, containment, overlap, near, and possible-occlusion
  relations; and
- composite, shape, layer-key, mask, and checksum validation results.

The `.p.zl` is referenced and its recursive manifest is included, but its large
native reconstruction arrays are not copied into JSON. Those arrays remain
losslessly available through the retained `.p.zl`.

Without `--exclude-invalid`, the command finishes with a nonzero exit status if
any scene is invalid, and the invalid record remains in `scenes`. With
`--exclude-invalid`, invalid records are moved to the lightweight
`excluded_scenes` audit array and do not make the command fail.
