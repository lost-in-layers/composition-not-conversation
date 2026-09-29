#!/usr/bin/env python3
"""Export human-reviewed question shards as conversation-simulator samples.

The reviewed scene records hold one question per (scene, category) with its
shards, gold layers and the shard-to-layer alignment. The simulator
wants a flat list of samples whose shards may each carry images. This module
converts between the two and renders the RGBA instance layers onto an opaque
canvas so a vision model does not receive alpha-composited-on-black frames.

Variants
--------
text            full RGB up front, shards are text only.
                Serves FULL, CONCAT_Q, SHARDED_Q, RECAP_Q, SNOWBALL_Q.
aligned_vq      each shard carries the layers in `question_shard_support`.
misaligned_vq   the same layers, reassigned so each group lands on a shard
                that none of its layers support (violations recorded in meta
                when a layer is supported by every shard).
visual          full question in shard 1, later shards reveal one layer each.
                `--layer-order` controls EVIDENCE_FIRST_V / EVIDENCE_LAST_V.
grounding       single turn: full RGB *and* every retained layer, question in
                shard 1. Experiment A condition 2 — the model answers and names
                its supporting layer ids, so these samples ask for the
                layer-selection response format.
oracle_gold     single turn: only a gold minimal sufficient layer set, no full
                RGB. Experiment A condition 3, an upper-bound diagnostic.

Image references are written relative to the exported JSON file because
`multimodal.image_base_dir` resolves them against the dataset file's parent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SOURCE = (
    REPO_ROOT
    / "data/derived/mulan-human-review/reviewer_2/question-shards-reviewed.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/derived/lic"
DEFAULT_RENDER_DIR = REPO_ROOT / "data/derived/lic_renders"

TASK_NAME = "layered_vqa"
VARIANTS = ("text", "aligned_vq", "misaligned_vq", "visual", "grounding", "oracle_gold")
# Variants whose shards carry layer images keyed off `question_shard_support`.
LAYER_SHARD_VARIANTS = ("aligned_vq", "misaligned_vq")
# Experiment A: one turn, every image attached at sample level. Their single
# shard holds the whole question, so a `full` run and a `concat` run of these
# files are the same conversation.
SINGLE_TURN_VARIANTS = ("grounding", "oracle_gold")
LAYER_ORDERS = ("canonical", "evidence_first", "evidence_last")
FLATTEN_MODES = ("white", "background")
OPENER_STYLES = ("raw", "casual")

# Verbs that open the annotator-voice `intent` shards, mapped to plainer words
# so the first user turn reads like something a person would type.
_OPENER_VERBS = {
    "determine": "figure out",
    "identify": "identify",
    "find": "find",
    "count": "count",
    "describe": "describe",
    "name": "name",
    "report": "say",
    "state": "say",
    "read": "read",
    "locate": "locate",
    "check": "check",
    "decide": "work out",
    "establish": "work out",
    "assess": "work out",
}

VISUAL_REVEAL_TEXT = "here's another part of the image"


class ExportError(RuntimeError):
    """Raised when a scene cannot be converted into a usable sample."""


# --------------------------------------------------------------------------
# scene helpers
# --------------------------------------------------------------------------


@dataclass
class LayerRef:
    """One numbered layer of a scene and where its pixels live."""

    layer_id: int
    source_path: Path
    is_background: bool
    label: str | None = None

    @property
    def display(self) -> str:
        """How a layer is named in the prompt.

        Deliberately identity-only. The reviewed `label` ("computer mouse")
        is frequently the answer itself, so putting it in the prompt would let
        a model answer from the text alone; it stays in `meta` for analysis.
        The background is called out because that is structural, not semantic.
        """
        if self.is_background:
            return f"layer {self.layer_id} (the background layer)"
        return f"layer {self.layer_id}"


@dataclass
class SceneIndex:
    """Layer lookup for a single scene, keyed by the ids questions refer to."""

    scene_id: str
    full_image: Path
    layers: dict[int, LayerRef] = field(default_factory=dict)

    def resolve(self, layer_ids: Iterable[int]) -> list[LayerRef]:
        refs = []
        for layer_id in layer_ids:
            ref = self.layers.get(int(layer_id))
            if ref is None:
                # Excluded layers stay referenced in some evidence notes; a
                # question that depends on one cannot be exported honestly.
                raise ExportError(
                    f"{self.scene_id}: layer {layer_id} is not a retained layer"
                )
            refs.append(ref)
        return refs

    @property
    def ordered_layer_ids(self) -> list[int]:
        return sorted(self.layers)


def index_scene(scene: dict[str, Any]) -> SceneIndex:
    scene_id = scene["scene_id"]
    index = SceneIndex(
        scene_id=scene_id,
        full_image=REPO_ROOT / scene["full_path"],
    )

    background = scene.get("background")
    if background:
        index.layers[int(background["layer_id"])] = LayerRef(
            layer_id=int(background["layer_id"]),
            source_path=REPO_ROOT / background["path"],
            is_background=True,
        )

    for layer in scene.get("layers", []):
        index.layers[int(layer["layer_id"])] = LayerRef(
            layer_id=int(layer["layer_id"]),
            source_path=REPO_ROOT / layer["path"],
            is_background=False,
            label=layer.get("label"),
        )

    return index


def iter_questions(scenes: list[dict[str, Any]]) -> Iterable[tuple[dict, dict]]:
    for scene in scenes:
        for question in scene.get("questions", []):
            yield scene, question


# --------------------------------------------------------------------------
# ordering
# --------------------------------------------------------------------------


def dependency_order(
    layer_ids: list[int], dependencies: list[dict[str, Any]]
) -> list[int]:
    """Order gold layers so every `before` precedes its `after`.

    `evidence_dependencies` entries read {"before": X, "after": Y}: X must be
    resolved before Y. Falls back to the given order when the constraints are
    cyclic or reference layers outside the set.
    """
    remaining = list(layer_ids)
    blockers: dict[int, set[int]] = {layer_id: set() for layer_id in remaining}
    for dep in dependencies:
        before, after = int(dep["before"]), int(dep["after"])
        if before in blockers and after in blockers:
            blockers[after].add(before)

    ordered: list[int] = []
    while remaining:
        ready = [lid for lid in remaining if not blockers[lid] - set(ordered)]
        if not ready:
            # Cyclic or unsatisfiable; keep the annotated order for the rest.
            ordered.extend(remaining)
            break
        pick = ready[0]
        ordered.append(pick)
        remaining.remove(pick)
    return ordered


def visual_reveal_order(
    question: dict[str, Any], index: SceneIndex, layer_order: str
) -> list[int]:
    """Which layers arrive, in which order, for the `visual` variant."""
    gold = [int(x) for x in question.get("gold_layer_ids", [])]
    gold = dependency_order(gold, question.get("evidence_dependencies", []))
    distractors = [
        int(x)
        for x in question.get("distractor_layer_ids", [])
        if int(x) in index.layers
    ]
    # A handful of questions list neither a gold nor a distractor role for a
    # retained layer. Treating those as distractors keeps every ordering of a
    # question delivering the same layer set, so only arrival order differs.
    unlabelled = [
        layer_id
        for layer_id in index.ordered_layer_ids
        if layer_id not in gold and layer_id not in distractors
    ]
    distractors = distractors + unlabelled

    if layer_order == "evidence_first":
        return gold + distractors
    if layer_order == "evidence_last":
        return distractors + gold
    return index.ordered_layer_ids


def oracle_layer_ids(
    question: dict[str, Any], index: SceneIndex
) -> tuple[list[int], str]:
    """The gold minimal sufficient set the oracle condition delivers.

    README-EXP defines the condition as "only a gold minimal sufficient layer
    set", so the annotated `minimal_sufficient_layer_sets` are authoritative and
    `gold_layer_ids` is only the fallback for a question that lists none. The
    background layer is *not* added: this is the upper-bound diagnostic, and
    padding it with scene context would make it a different measurement.

    On the current 300-question annotation the first minimal set equals
    `gold_layer_ids` everywhere, so the distinction is definitional today; it
    stops mattering silently if a later review adds a smaller sufficient set.
    """
    dependencies = question.get("evidence_dependencies", [])

    for candidate in question.get("minimal_sufficient_layer_sets") or []:
        layer_ids = [int(x) for x in candidate]
        if layer_ids and all(layer_id in index.layers for layer_id in layer_ids):
            return dependency_order(layer_ids, dependencies), "minimal_sufficient"

    gold = [int(x) for x in question.get("gold_layer_ids", [])]
    if gold and all(layer_id in index.layers for layer_id in gold):
        return dependency_order(gold, dependencies), "gold_layer_ids"

    # No sufficient set survives the retained layers, so there is no honest
    # oracle input for this question.
    raise ExportError(
        f"{index.scene_id}: no sufficient layer set resolves to retained layers "
        f"for {question['question_id']}"
    )


def aligned_schedule(
    shards: list[dict[str, Any]], support: dict[int, list[int]]
) -> dict[int, list[int]]:
    """Map each shard to the supporting layers that are new at that turn.

    A layer supporting several shards arrives once, at its earliest shard, so
    later shards that depend on it introduce no new image.
    """
    revealed: set[int] = set()
    schedule: dict[int, list[int]] = {}
    for shard in shards:
        shard_id = int(shard["shard_id"])
        new_ids = [lid for lid in support.get(shard_id, []) if lid not in revealed]
        revealed.update(new_ids)
        schedule[shard_id] = new_ids
    return schedule


def misaligned_schedule(
    aligned: dict[int, list[int]], support: dict[int, list[int]]
) -> tuple[dict[int, list[int]], list[tuple[int, int]]]:
    """Deliver the same layers, but each on a turn whose shard they do not support.

    This keeps the condition information-matched to the aligned one: identical
    layers, identical per-turn image counts, identical final evidence set. Only
    the pairing between a question shard and its supporting layer is broken, so
    the contrast isolates alignment rather than confounding it with how much
    evidence arrived.

    Misalignment is checked against `question_shard_support`, not against the
    aligned schedule: a layer that supports several shards must not land on any
    of them (rotating the schedule by one slot, the previous approach, parked
    ~95% of layers on the *next* shard they support). Layer groups move whole,
    so the per-turn image-count multiset is preserved exactly.

    Some questions have a layer that every shard supports, making strict
    misalignment impossible. The permutation minimizing the number of
    (layer, supporting shard) pairings is used and those residual pairings are
    returned so they can be recorded in `meta` and excluded from analysis.
    """
    slots = list(aligned)
    nonempty = [shard_id for shard_id in slots if aligned[shard_id]]
    if not nonempty:
        return dict(aligned), []

    support_sets = {shard_id: set(support.get(shard_id, [])) for shard_id in slots}

    def violations(shard_id: int, group: list[int]) -> list[tuple[int, int]]:
        return [(shard_id, lid) for lid in group if lid in support_sets[shard_id]]

    # Assign the most constrained groups first; try landing slots in shard
    # order so results are deterministic. Group counts are tiny (<= 4 groups
    # over <= 7 shards), so exhaustive search with a best-so-far bound is fine.
    order = sorted(nonempty, key=lambda s: (-len(aligned[s]), s))
    best: tuple[int, dict[int, list[int]], list[tuple[int, int]]] | None = None

    def assign(
        index: int,
        used: set[int],
        placed: dict[int, list[int]],
        broken: list[tuple[int, int]],
    ) -> None:
        nonlocal best
        if best is not None and len(broken) >= best[0]:
            return  # cannot improve on the best assignment found so far
        if index == len(order):
            best = (len(broken), dict(placed), list(broken))
            return
        group = aligned[order[index]]
        for slot in slots:
            if slot in used:
                continue
            placed[slot] = group
            assign(index + 1, used | {slot}, placed, broken + violations(slot, group))
            del placed[slot]

    assign(0, set(), {}, [])
    assert best is not None  # some permutation always exists
    _, placed, broken = best

    schedule = {shard_id: placed.get(shard_id, []) for shard_id in slots}
    return schedule, sorted(broken)


def demand_level(question: dict[str, Any]) -> int:
    """Heuristic evidence-demand level (1-5) per README-EXP section 3.

    This is a starting stratification, not an annotation. Override it with a
    curated column before making cliff claims in the paper.
    """
    n_gold = len(question.get("gold_layer_ids", []))
    n_distractors = len(question.get("distractor_layer_ids", []))
    n_hops = len(question.get("evidence_dependencies", []))
    is_occlusion = question.get("category") == "occlusion"

    if n_gold <= 1:
        return 1 if n_distractors == 0 else 2
    if n_gold >= 4 or (n_gold >= 3 and n_distractors >= 2):
        return 5
    if n_hops >= 1 or is_occlusion or n_gold >= 3:
        return 4
    return 3


# --------------------------------------------------------------------------
# opener rewriting
# --------------------------------------------------------------------------


def casual_opener(text: str) -> str:
    """Rewrite an annotator-voice intent shard into a user-voice question.

    `user_agent.py` sends shard 1 verbatim, without LLM rephrasing, so this is
    the only turn whose wording we control directly.
    """
    body = text.strip().rstrip(".").strip()
    if not body:
        return "can you help me with this image?"

    words = body.split()
    lead = words[0].lower()
    if lead in _OPENER_VERBS:
        words = [_OPENER_VERBS[lead]] + words[1:]
    body = " ".join(words)
    body = body.replace(" in the image", " in this image")
    body = body.replace(" in the picture", " in this picture")
    body = body[0].lower() + body[1:]
    return f"can you help me {body}?"


# --------------------------------------------------------------------------
# layer rendering
# --------------------------------------------------------------------------


def render_layers(
    scenes: list[dict[str, Any]],
    render_dir: Path,
    flatten_mode: str,
    overwrite: bool,
) -> dict[str, int]:
    """Flatten every RGBA layer onto an opaque canvas.

    Instance layers are stored as RGBA with real transparency. Most vision
    APIs drop or blacken alpha, so a raw layer arrives as a near-black frame.
    """
    from PIL import Image  # imported lazily so the exporter stays importable

    counts = {"written": 0, "skipped": 0}

    for scene in scenes:
        index = index_scene(scene)
        target_dir = render_dir / index.scene_id
        target_dir.mkdir(parents=True, exist_ok=True)

        background_rgba = None
        if flatten_mode == "background":
            background_ref = next(
                (ref for ref in index.layers.values() if ref.is_background), None
            )
            if background_ref is not None:
                with Image.open(background_ref.source_path) as handle:
                    background_rgba = handle.convert("RGBA")

        for ref in index.layers.values():
            output = target_dir / f"layer_{ref.layer_id}.png"
            if output.exists() and not overwrite:
                counts["skipped"] += 1
                continue

            with Image.open(ref.source_path) as handle:
                layer = handle.convert("RGBA")

            if ref.is_background or background_rgba is None:
                canvas = Image.new("RGBA", layer.size, (255, 255, 255, 255))
            else:
                canvas = background_rgba.copy()

            flattened = Image.alpha_composite(canvas, layer).convert("RGB")
            flattened.save(output)
            counts["written"] += 1

    return counts


# --------------------------------------------------------------------------
# sample construction
# --------------------------------------------------------------------------


def image_ref(path: Path, base_dir: Path, label: str, detail: str) -> dict[str, Any]:
    """An image reference relative to the dataset file, carrying a label.

    `multimodal.image_content_part` reads only `path`/`url`/`detail`, so the
    extra `label` key rides along for prompt construction and never reaches
    the API payload.
    """
    return {
        "path": os.path.relpath(path, base_dir).replace(os.sep, "/"),
        "detail": detail,
        "label": label,
    }


def layer_image_refs(
    refs: list[LayerRef],
    scene_id: str,
    render_dir: Path,
    base_dir: Path,
    detail: str,
) -> list[dict[str, Any]]:
    return [
        image_ref(
            render_dir / scene_id / f"layer_{ref.layer_id}.png",
            base_dir,
            ref.display,
            detail,
        )
        for ref in refs
    ]


def build_sample(
    scene: dict[str, Any],
    question: dict[str, Any],
    *,
    variant: str,
    layer_order: str,
    opener_style: str,
    include_full_image: bool,
    render_dir: Path,
    base_dir: Path,
    detail: str,
) -> dict[str, Any]:
    index = index_scene(scene)
    shards = question["shards"]
    support = {int(k): [int(v) for v in vs] for k, vs in question["question_shard_support"].items()}
    gold_ids = [int(x) for x in question.get("gold_layer_ids", [])]
    distractor_ids = [
        int(x) for x in question.get("distractor_layer_ids", []) if int(x) in index.layers
    ]

    sample_images: list[dict[str, Any]] = []
    if include_full_image:
        sample_images.append(
            image_ref(index.full_image, base_dir, "the full image", detail)
        )
    # What a recap turn re-sends: the full RGB, whatever the variant delivered
    # per shard. RECAP_V is defined as "provide the full RGB image again", and
    # re-attaching every layer render instead would change the condition.
    recap_images = [image_ref(index.full_image, base_dir, "the full image", detail)]

    out_shards: list[dict[str, Any]] = []
    reveal_order: list[int] = []
    misaligned_pairs: list[tuple[int, int]] = []
    schedule: dict[int, list[int]] = {}
    oracle_ids: list[int] | None = None
    oracle_source: str | None = None

    if variant in SINGLE_TURN_VARIANTS:
        # One turn holding the whole question, with every image at sample level:
        # `all_sample_images` lists sample images before shard images, so the
        # prompt manifest names them in the order they are attached.
        out_shards.append({"shard_id": 1, "shard": question["question"]})
        if variant == "grounding":
            layer_ids = index.ordered_layer_ids
        else:
            layer_ids, oracle_source = oracle_layer_ids(question, index)
            oracle_ids = list(layer_ids)
        sample_images.extend(
            layer_image_refs(
                index.resolve(layer_ids), index.scene_id, render_dir, base_dir, detail
            )
        )
        reveal_order = list(layer_ids)
    elif variant == "visual":
        opener = question["question"]
        out_shards.append({"shard_id": 1, "shard": opener})
        reveal_order = visual_reveal_order(question, index, layer_order)
        for position, layer_id in enumerate(reveal_order, start=2):
            ref = index.resolve([layer_id])[0]
            out_shards.append(
                {
                    "shard_id": position,
                    "shard": VISUAL_REVEAL_TEXT,
                    "images": layer_image_refs(
                        [ref], index.scene_id, render_dir, base_dir, detail
                    ),
                }
            )
    else:
        if variant in LAYER_SHARD_VARIANTS:
            schedule = aligned_schedule(shards, support)
            if variant == "misaligned_vq":
                schedule, misaligned_pairs = misaligned_schedule(schedule, support)

        for position, shard in enumerate(shards):
            shard_id = int(shard["shard_id"])
            text = shard["text"]
            if position == 0 and opener_style == "casual":
                text = casual_opener(text)

            out_shard: dict[str, Any] = {"shard_id": shard_id, "shard": text}

            new_ids = schedule.get(shard_id, [])
            if new_ids:
                reveal_order.extend(new_ids)
                out_shard["images"] = layer_image_refs(
                    index.resolve(new_ids),
                    index.scene_id,
                    render_dir,
                    base_dir,
                    detail,
                )

            out_shards.append(out_shard)

    return {
        "task_id": question["question_id"],
        "task": TASK_NAME,
        "question": question["question"],
        "answer": question["answer"],
        "images": sample_images,
        "recap_images": recap_images,
        "shards": out_shards,
        "meta": {
            "scene_id": scene["scene_id"],
            "subset": scene.get("subset"),
            "category": question["category"],
            "variant": variant,
            "layer_order": layer_order if variant == "visual" else None,
            # What the assistant is asked to produce. `answer_and_layers` makes
            # the task load the layer-selection prompt and score the predicted
            # evidence set; every other variant asks for the answer alone.
            "response_format": (
                "answer_and_layers" if variant == "grounding" else "answer"
            ),
            "gold_layer_ids": gold_ids,
            "distractor_layer_ids": distractor_ids,
            # Which ids a model could legitimately cite, and which of them is
            # the background — scoring needs both, on every variant.
            "candidate_layer_ids": index.ordered_layer_ids,
            "background_layer_id": next(
                (
                    layer_id
                    for layer_id, ref in sorted(index.layers.items())
                    if ref.is_background
                ),
                None,
            ),
            # The evidence set the oracle condition actually delivered.
            "oracle_layer_ids": oracle_ids,
            "oracle_source": oracle_source,
            "layer_roles": question.get("layer_roles", {}),
            "layer_labels": {
                str(layer_id): ref.label
                for layer_id, ref in index.layers.items()
                if ref.label
            },
            "minimal_sufficient_layer_sets": question.get(
                "minimal_sufficient_layer_sets", []
            ),
            "evidence_dependencies": question.get("evidence_dependencies", []),
            "question_shard_support": question["question_shard_support"],
            "shard_roles": {
                str(shard["shard_id"]): shard["role"] for shard in shards
            },
            "reveal_order": reveal_order,
            "aligned_schedule": {
                str(k): v for k, v in aligned_schedule(shards, support).items()
            },
            # Only for misaligned_vq: the schedule actually delivered, plus any
            # (shard_id, layer_id) pairings that could not be broken because
            # every shard supports that layer. Exclude those from contrasts.
            "misaligned_schedule": (
                {str(k): v for k, v in schedule.items()}
                if variant == "misaligned_vq"
                else None
            ),
            "misaligned_violations": (
                [list(pair) for pair in misaligned_pairs]
                if variant == "misaligned_vq"
                else None
            ),
            "demand_level": demand_level(question),
            "shard_count": len(out_shards),
            "intent_shard_raw": shards[0]["text"],
            "confidence": question.get("confidence"),
            "human_review": question.get("human_review", {}),
        },
    }


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def export(
    scenes: list[dict[str, Any]],
    *,
    variant: str,
    layer_order: str,
    opener_style: str,
    include_full_image: bool,
    render_dir: Path,
    output_path: Path,
    detail: str,
    limit: int | None = None,
    categories: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    base_dir = output_path.parent
    samples: list[dict[str, Any]] = []
    skipped: list[str] = []

    for scene, question in iter_questions(scenes):
        if categories and question["category"] not in categories:
            continue
        try:
            samples.append(
                build_sample(
                    scene,
                    question,
                    variant=variant,
                    layer_order=layer_order,
                    opener_style=opener_style,
                    include_full_image=include_full_image,
                    render_dir=render_dir,
                    base_dir=base_dir,
                    detail=detail,
                )
            )
        except ExportError as error:
            skipped.append(str(error))
            continue
        if limit and len(samples) >= limit:
            break

    return samples, skipped


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--render-dir", type=Path, default=DEFAULT_RENDER_DIR)
    parser.add_argument("--variant", choices=VARIANTS, default="text")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export every variant and every visual layer order in one pass.",
    )
    parser.add_argument("--layer-order", choices=LAYER_ORDERS, default="canonical")
    parser.add_argument("--opener-style", choices=OPENER_STYLES, default="casual")
    parser.add_argument("--flatten-mode", choices=FLATTEN_MODES, default="white")
    parser.add_argument("--detail", choices=("auto", "low", "high"), default="high")
    parser.add_argument(
        "--include-full-image",
        dest="include_full_image",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Default: on for the text and grounding variants, off for the other layer variants.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--overwrite-renders", action="store_true")
    return parser.parse_args(argv)


def export_one(
    scenes: list[dict[str, Any]],
    args: argparse.Namespace,
    variant: str,
    layer_order: str,
) -> int:
    include_full_image = args.include_full_image
    if include_full_image is None:
        # `grounding` is defined as full image *plus* all layers; `oracle_gold`
        # is defined as the gold set alone, so it must not get the full RGB.
        include_full_image = variant in ("text", "grounding")

    if args.output is not None:
        output_path = args.output
    else:
        stem = f"visual_{layer_order}" if variant == "visual" else variant
        output_path = args.output_dir / f"layered_vqa_{stem}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples, skipped = export(
        scenes,
        variant=variant,
        layer_order=layer_order,
        opener_style=args.opener_style,
        include_full_image=include_full_image,
        render_dir=args.render_dir,
        output_path=output_path,
        detail=args.detail,
        limit=args.limit,
        categories=set(args.categories) if args.categories else None,
    )

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(samples, handle, indent=2)

    manifest = {
        "source": str(args.source),
        "variant": variant,
        "layer_order": layer_order if variant == "visual" else None,
        "opener_style": args.opener_style,
        "flatten_mode": args.flatten_mode,
        "include_full_image": include_full_image,
        "detail": args.detail,
        "sample_count": len(samples),
        "skipped": skipped,
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    note = f" ({len(skipped)} skipped)" if skipped else ""
    print(f"wrote {len(samples)} samples{note} -> {output_path}")
    return len(samples)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    with args.source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    scenes = payload["scenes"]

    if args.all:
        jobs = [(variant, "canonical") for variant in VARIANTS if variant != "visual"]
        jobs += [("visual", order) for order in LAYER_ORDERS]
        # `layer_order` is meaningless for the single-turn variants; "canonical"
        # above is a placeholder and is written to their manifests as None.
        if args.output is not None:
            raise SystemExit("--output cannot be combined with --all")
    else:
        jobs = [(args.variant, args.layer_order)]

    if not args.skip_render and any(variant != "text" for variant, _ in jobs):
        counts = render_layers(
            scenes, args.render_dir, args.flatten_mode, args.overwrite_renders
        )
        print(
            f"renders: {counts['written']} written, {counts['skipped']} already present"
        )

    for variant, layer_order in jobs:
        export_one(scenes, args, variant, layer_order)
    return 0


if __name__ == "__main__":
    sys.exit(main())
