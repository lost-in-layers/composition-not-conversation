# MuLAn LEAF-VQA Pilot Pipeline

This document supersedes the earlier FLUX/Qwen-Image-Layered synthetic-data
pilot. The current source is the released MuLAn v1.0 annotation set and the
pre-rendered LVIS/COCO and LAION layer stacks.

The immediate goal is a **600-example gold pilot**:

- 6 question-program categories
- 100 accepted scene-question programs per category
- category-specific easy/medium/hard coverage rather than forcing every
  category into incompatible global evidence-demand bins
- one canonical question, answer, minimal evidence-layer set, and valid shard
  set per program
- all 600 manually reviewed; at least 120 independently double-annotated

An image is not assigned permanently to a question category. A scene may
support several candidate question programs. The program operator and evidence
requirements determine the category. Prefer one accepted program per scene in
the pilot so that categories remain image-disjoint.

## 1. Available Inputs

The local inventory files are:

```text
data/pre-data/inventories/lvis.csv
data/pre-data/inventories/laion.csv
data/pre-data/stats/lvis/
data/pre-data/stats/laion/
```

The inventories contain image IDs, source-image references, layer counts,
render paths, and image-level COCO category names where available. They do not
contain a complete scene description or per-layer semantic mapping. In
particular, `categories` in `lvis.csv` was built from COCO instance JSON and is
an image-level set, not a claim that `layer_1` has the first listed category.
The LAION inventory has no corresponding category source, so that column is
empty.

The current render archives are hosted at:

```text
https://huggingface.co/lost-in-layer/Layered-VQA/tree/main/renders
  renders/lvis.zip
  renders/laion.zip
```

Each successful rendered scene contains:

```text
<subset>/<image_id>/
  full.png       # flattened RGB composite
  layer_0.png    # reconstructed RGB background
  layer_1.png    # full-canvas RGBA instance
  ...
  layer_N.png
```

The released MuLAn `.p.zl` files are the richer native annotations. The local
inventory references them, but they are not currently present in this Git
workspace. Keep or re-extract them when building scene records; rendered PNGs
alone omit useful captions and reconstruction fields.

## 2. What Is Native and What Must Be Derived

### Native in each `.p.zl`

The documented MuLAn annotation structure includes:

```text
captioning
image.blip2
image.llava
background.blip2
background.llava
background.original_image_mask
background.inpainted_delta
instances[k].blip2
instances[k].original_image_mask
instances[k].instance_mask
instances[k].inpainted_delta
instances[k].instance_alpha
```

The official `dataset_decomposition.py` exports layer paths and BLIP-2/LLaVA
captions to `meta_data.csv`. The enriched inventory implementation in
`src/vecgen/mulan_dataset_tool.py` now exports those captions, every native
scalar/text field, a recursive descriptor and SHA-256 for every array/binary
field, and the SHA-256 of the canonical compressed `.p.zl`.

### Recoverable from rendered PNGs

For each RGBA instance layer, its alpha channel is the practical layer mask:

```text
A_i = alpha(layer_i) / 255
M_i = A_i >= alpha_threshold
```

From `M_i`, compute rather than annotate:

- visible bounding box
- alpha-weighted centroid
- visible area and image-area fraction
- connected-component count
- border contact and truncation flags
- pairwise intersection and overlap ratios

### Stack ordering

MuLAn constructs an ordered RGBA stack from the background/farthest content to
the nearest/top content. In the current renderer, `layer_0` is the background
and numerically ordered instance layers are alpha-composited in increasing
order. Store this as `z_index`, not as metric depth. A larger `z_index` is
treated as being on top only after verifying that flattening the ordered stack
reconstructs `full.png`.

Do not claim front/behind merely from two non-overlapping layer indices. Use
stack order for a depth or occlusion relation only when the two masks or their
amodal supports overlap meaningfully, and manually verify pilot examples.

### Derived rather than native

The following scene-record fields must be calculated or verified:

- `bbox`, `centroid`, `area_fraction`
- `left_of`, `right_of`, `above`, `below`, `near`
- `overlaps`, `contains`, `possible_occludes`
- dominant color and coarse size attributes
- per-layer canonical label
- decomposition-quality flags

The released schema does not document a numeric depth map or a ready-made
occlusion graph in every `.p.zl`. MuLAn used depth and occlusion models to
construct its stack, but LEAF should treat the released stack order as ordinal
evidence and derive conservative pairwise relations.

## 3. Restoring Captions and Labels

Use the following label priority:

1. Native `instances[k].blip2` caption from the matching `.p.zl`.
2. For COCO/LVIS, match the layer alpha mask to COCO instance masks by IoU and
   inherit the matched instance category only above a conservative threshold.
3. Use the image-level `categories` inventory value only as a candidate label
   list, never as a per-layer mapping.
4. For unmatched COCO layers and LAION layers, run a VLM over both the isolated
   full-canvas layer and the full composite to propose a short noun phrase.
5. Human-verify every label used in a gold question.

COCO caption JSON may supply human-written image-level descriptions, but it
does not replace per-layer captions. The MuLAn LAION file-to-URL map contains
only `file_id` and `laion_url`; it does not include original LAION alt text.
Native MuLAn BLIP-2 captions are therefore the first recovery target for both
subsets.

The enriched CSV includes:

```text
annotation_sha256
annotation_size_bytes
annotation_captioning_json
annotation_metadata_json
image_blip2
image_llava
background_blip2
background_llava
instance_keys_json
instance_blip2_json
```

Large arrays remain losslessly in `.p.zl`; their CSV manifest entries record
type, shape, dtype, size, and SHA-256 rather than duplicating binary bytes in a
CSV cell. Use `mulan_dataset_tool verify` to confirm that every referenced
`.p.zl` still matches its inventory checksum.

## 4. Scene Record

Create one scene record per rendered stack:

```json
{
  "scene_id": "lvis_000000000086",
  "subset": "lvis",
  "image_id": "000000000086",
  "full_path": ".../full.png",
  "image_caption_blip2": "...",
  "image_caption_llava": "...",
  "quality": "gold",
  "layers": [
    {
      "layer_id": 0,
      "role": "background",
      "path": ".../layer_0.png",
      "caption_blip2": "..."
    },
    {
      "layer_id": 1,
      "role": "instance",
      "path": ".../layer_1.png",
      "label": "person",
      "label_source": "mulan_blip2_human_verified",
      "bbox_xyxy": [42, 18, 381, 511],
      "centroid_xy": [204.2, 267.8],
      "area_fraction": 0.19,
      "z_index": 1,
      "attributes": {"dominant_color": "blue"},
      "quality_flags": []
    }
  ],
  "relations": [
    {
      "subject": 2,
      "predicate": "left_of",
      "object": 3,
      "score": 0.94,
      "source": "geometry"
    }
  ]
}
```

### Conservative geometry rules

Use normalized image coordinates and retain continuous scores. Convert a score
to a discrete relation only above a threshold fixed on pilot-development data.

- `left_of(i,j)`: the x-separation is sufficiently larger than the uncertainty
  margin; centroid order alone is not enough for strongly overlapping objects.
- `above(i,j)`: analogous y-separation rule.
- `near(i,j)`: normalized boundary or centroid distance below a fixed
  threshold; do not mix definitions after annotation starts.
- `overlaps(i,j)`: alpha-mask intersection or amodal-support intersection
  exceeds a minimum pixel and ratio threshold.
- `possible_occludes(i,j)`: `i` has higher `z_index` than `j` and their supports
  overlap. This remains a candidate until visually verified.
- `between(i,j,k)`: `i` lies near the line segment joining `j` and `k`, not
  merely between their x coordinates.

Save the raw scores so threshold sensitivity can be evaluated later.

## 5. Question-Program Categories and Eligibility

Category is assigned to a **question program**, not an image or layer. Generate
all eligible candidate programs for a scene, reject ambiguous programs, and
then sample accepted programs to fill the balanced category buckets.

### A. Direct perception

Eligibility:

- at least one coherent instance layer
- human-verified label or stable coarse attribute
- target is visible enough to recognize

Programs:

```text
QUERY_LABEL(target)
QUERY_COLOR(target)
```

### B. Attribute binding

Eligibility:

- at least two candidate objects
- target has a verified attribute
- an anchor relation identifies exactly one target

Program:

```text
QUERY_COLOR(FILTER(left_of, anchor=bicycle))
```

### C. Spatial/topological reasoning

Eligibility:

- at least two layers for pairwise relations; at least three for `between`
- relation passes conservative geometry thresholds
- answer is unique under the stored graph

Programs:

```text
QUERY_OBJECT(left_of(anchor))
QUERY_OBJECT(between(anchor_1, anchor_2))
```

### D. Occlusion/depth

Eligibility:

- at least two coherent layers
- meaningful support overlap
- validated stack order and visually clear front/behind relationship

Programs:

```text
QUERY_FRONT_OBJECT(overlap_pair)
QUERY_REVEALED_OBJECT(remove=front_layer)
```

### E. Counting/set reasoning

Eligibility:

- at least two verified foreground instances
- all objects in the counted set are represented by separate usable layers
- query does not silently count unextracted objects fused into the background

Programs:

```text
COUNT(FILTER(right_of, anchor))
COMPARE_COUNT(set_1, set_2)
```

### F. Multi-hop reference

Eligibility:

- at least three coherent layers
- a graph path of two or three relations
- every intermediate reference and the final target are unique

Program:

```text
QUERY_OBJECT(behind(nearest_to(person)))
```

One scene can produce candidates in multiple categories. For the pilot, select
one accepted category/program per scene where possible. If a scene contributes
two questions, keep both in the same dataset split and use scene-level
bootstrap confidence intervals.

## 6. Category Difficulty and Evidence Demand

For the 100 examples in each question category, target approximately:

- 30 category-relative easy examples
- 40 category-relative medium examples
- 30 category-relative hard examples

Define these strata separately for each category. For example, a hard direct
perception question may still need only one target layer but contain several
similar distractors, while even an easy multi-hop question necessarily uses
multiple evidence units.

Separately, assign every accepted program a global evidence-demand descriptor:

| Level | Construction |
|---:|---|
| 1 | one necessary layer, no meaningful distractor |
| 2 | one necessary layer plus one or two distractors |
| 3 | two necessary evidence units or one explicit relation |
| 4 | two to three units, occlusion, or a two-hop reference |
| 5 | three to four units plus similar distractors, small evidence, or controlled decomposition noise |

Difficulty is attached to the program, not copied directly from raw layer
count. A five-layer image can still yield a Level-2 question if only one target
and one distractor matter.

Do not require 20 examples from every category at every global level: several
cells are structurally impossible and forcing them would corrupt the task. For
the later Visual Evidence Cliff experiment, construct a separate controlled
slice balanced across global demand levels, and compare demand within compatible
program templates. This keeps task-family effects separate from evidence-demand
effects.

## 7. Question and Evidence Generation

For each candidate:

1. Execute a symbolic program over the scene record.
2. Compute its unique answer.
3. Compute the minimal required layer set.
4. Reject programs with zero or multiple valid answers.
5. Render a canonical template question.
6. Optionally ask a language model to paraphrase without changing the program.
7. Re-parse or verify the paraphrase against the symbolic program.
8. Human-review question, answer, evidence set, and layer quality.

Do not ask a VLM to invent the answer or gold evidence. A VLM may propose
labels, attributes, paraphrases, and shards, but deterministic program execution
and human review are the authorities.

Canonical QA record:

```json
{
  "example_id": "leaf_pilot_000001",
  "scene_id": "lvis_000000000086",
  "category": "attribute_binding",
  "difficulty": 3,
  "program": [
    {"op": "select", "label": "bicycle"},
    {"op": "relate", "predicate": "left_of"},
    {"op": "query", "field": "dominant_color"}
  ],
  "question": "What color is the object to the left of the bicycle?",
  "answer": "red",
  "gold_layer_ids": [2, 4],
  "answer_source": "program",
  "review_status": "accepted"
}
```

## 8. Question and Visual Sharding

Create paired presentations from the same accepted QA record:

- `FULL`: full composite plus canonical question
- `CONCAT_Q`: full composite plus all question shards in one turn
- `SHARDED_Q`: full composite plus one textual constraint per turn
- `CONCAT_V`: full question plus all full-canvas layer cards together
- `SHARDED_V`: full question plus one full-canvas layer per turn
- `SHARDED_VQ`: question constraints and full-canvas visual layers both arrive
  incrementally
- `RECAP`: sharded interaction followed by full question and full image
- `SNOWBALL`: each turn repeats accumulated information and adds one shard

Question shards must preserve all answer-relevant information, state the intent
first, be order-independent after the first shard, contain one atomic constraint
each, and minimally transform the canonical wording.

For the short human-authored VQA questions, use adaptive atomic sharding rather
than the source paper's three-segment admission threshold: allow two shards for
a genuine intent-plus-target question, use three or four only when distinct
constraints exist, and never pad a question with redundant turns. The generator
and category-specific prompts are documented in
[`docs/question_sharding.md`](question_sharding.md).

Example:

```text
FULL:
  What color is the partly occluded object to the left of the bicycle?

SHARDED_Q:
  1. Tell me the color of a particular object in this image.
  2. The target object is located to the left of the bicycle.
  3. The target object is partly occluded.
```

Visual shards must retain the original full canvas and coordinates. Do not crop
each object to a different size, and do not add semantic labels in the principal
visual-only condition.

## 9. MuLAn Scene-Record Output

```text
data/derived/mulan-scenes/
  lvis_scenes.json
  laion_scenes.json
```

Build both files with:

```bash
python -m src.vecgen.build_mulan_scene_records --workers 8
```

The JSON records reference the alpha channels already stored in the RGBA layer
PNGs; they do not duplicate masks as RLE or separate PNG files. See
`docs/mulan_scene_records.md` for the complete schema and validation behavior.

Question-program and review artifacts can remain separate downstream outputs:

```text
data/leaf_pilot/
  candidates/
    direct_perception.jsonl
    attribute_binding.jsonl
    spatial_topological.jsonl
    occlusion_depth.jsonl
    counting_set.jsonl
    multi_hop_reference.jsonl
  questions/
    proposed.jsonl
    accepted.jsonl
  shards/
    question_shards.jsonl
    visual_orders.jsonl
  reviews/
    primary.jsonl
    double_annotation.jsonl
  splits/
    pilot_dev.json
    pilot_test.json
```

## 10. Existing Inventory and Render Commands

The existing tool can rebuild inventories and renders once native `.p.zl` files
and source images are available:

```powershell
python -m src.vecgen.mulan_dataset_tool inventory `
  --subset lvis `
  --pzl-dir <mulan-lvis-pzl-dir> `
  --images-dir <coco-2017-dir> `
  --coco-annotations-dir <coco-annotations-dir> `
  --inventory data/pre-data/inventories/lvis.csv

python -m src.vecgen.mulan_dataset_tool inventory `
  --subset laion_aesthetic_v2_6.5 `
  --pzl-dir <mulan-laion-pzl-dir> `
  --url-map <file_id_url_linking.csv> `
  --inventory data/pre-data/inventories/laion.csv

python -m src.vecgen.mulan_dataset_tool stats `
  --inventory data/pre-data/inventories/lvis.csv `
  --output-dir data/pre-data/stats/lvis

python -m src.vecgen.mulan_dataset_tool render `
  --inventory data/pre-data/inventories/lvis.csv `
  --output-dir data/pre-data/renders/lvis
```

The complete Ubuntu download/extraction/verification/cleanup procedure is in
[`docs/mulan_bootstrap.md`](mulan_bootstrap.md). Scene-graph construction,
program generation, sharding, and review export are the next implementation
stages.

## 11. Acceptance Gates Before Scaling

Do not scale from 600 to 6,000 until the pilot achieves:

- at least 90% of accepted questions have a unique program-derived answer
- at least 95% human answer agreement
- at least 0.85 inter-annotator layer-set F1
- no material FULL-versus-CONCAT degradation caused by rephrasing
- stable geometry conclusions under small threshold perturbations
- clear SHARDED degradation beyond paraphrase and layer-quality noise
- all test scenes image-disjoint from development scenes

If occlusion/depth questions fail these gates, keep them as a smaller manually
verified diagnostic category rather than deriving them automatically from stack
index alone.
