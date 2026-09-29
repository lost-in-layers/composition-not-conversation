"""Acceptance validation, category eligibility, and curated scene creation.

Curated (accepted) scenes contain no null/empty required fields. Original
labels, captions and technical nulls are preserved untouched inside a
``provenance`` block; human values live in the active fields.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

CATEGORIES = (
    "direct_ask",
    "attribute_binding",
    "spatial",
    "occlusion",
    "counting",
    "multi_hop_reference",
)

CATEGORY_LABELS = {
    "direct_ask": "Direct Ask",
    "attribute_binding": "Attribute Binding",
    "spatial": "Spatial",
    "occlusion": "Occlusion",
    "counting": "Counting",
    "multi_hop_reference": "Multi-hop Reference",
}

EXCLUSION_REASONS = (
    "incorrect_segmentation",
    "fused_objects",
    "fragmented_object",
    "duplicate",
    "irrelevant_or_hallucinated",
    "unusable_transparency",
    "severe_artifact",
    "meaningless_layer",
    "other",
)


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _layer_review(review: Dict[str, Any], layer_id: Any) -> Dict[str, Any]:
    reviews = review.get("layer_reviews") or {}
    return reviews.get(str(layer_id)) or reviews.get(layer_id) or {}


def reviewed_questions(review: Dict[str, Any]) -> Dict[str, str]:
    """Return non-empty questions for currently selected categories.

    The legacy single ``human_question`` field is mapped to the primary
    category so drafts created by the earlier UI remain usable.
    """
    suitable = review.get("suitable_categories") or []
    raw = review.get("human_questions") or {}
    questions = {
        cat: value.strip()
        for cat, value in raw.items()
        if cat in suitable and cat in CATEGORIES and isinstance(value, str) and value.strip()
    }
    legacy = review.get("human_question")
    primary = review.get("primary_category")
    if not questions and primary in suitable and isinstance(legacy, str) and legacy.strip():
        questions[primary] = legacy.strip()
    return questions


def reviewed_answers(review: Dict[str, Any], questions: Dict[str, str]) -> Dict[str, str]:
    """Return non-empty answers corresponding to reviewed questions."""
    raw = review.get("human_answers") or {}
    return {
        cat: value.strip()
        for cat, value in raw.items()
        if cat in questions and isinstance(value, str) and value.strip()
    }


def retained_layer_ids(scene: Dict[str, Any], review: Dict[str, Any]) -> List[int]:
    retained = []
    for layer in scene.get("layers") or []:
        lr = _layer_review(review, layer.get("layer_id"))
        if lr.get("decision") == "keep":
            retained.append(layer["layer_id"])
    return retained


def _retained_relations(scene: Dict[str, Any], retained: Set[int], review: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Original relations restricted to retained pairs, minus human-removed,
    plus human-added ones."""
    removed = {tuple(r) for r in (review.get("relations_removed") or [])}
    result = []
    for rel in scene.get("relations") or []:
        key = (rel.get("subject_layer_id"), rel.get("predicate"), rel.get("object_layer_id"))
        if rel.get("subject_layer_id") in retained and rel.get("object_layer_id") in retained and key not in {tuple(k) for k in removed}:
            result.append(dict(rel))
    for added in review.get("relations_added") or []:
        subject = added.get("subject_layer_id")
        obj = added.get("object_layer_id")
        if subject in retained and obj in retained and not _is_blank(added.get("predicate")):
            result.append(
                {
                    "subject_layer_id": subject,
                    "predicate": added["predicate"],
                    "object_layer_id": obj,
                    "score": None,
                    "source": "human_review",
                }
            )
    return result


def category_eligibility(scene: Dict[str, Any], review: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Minimum eligibility per category, computed from retained layers only.

    These signals are advisory only. Every category remains selectable and the
    human reviewer's decision is authoritative.
    """
    retained = set(retained_layer_ids(scene, review))
    n = len(retained)
    relations = _retained_relations(scene, retained, review)
    directional = [r for r in relations if r.get("predicate") in ("left_of", "right_of", "above", "below", "near", "contains")]
    occlusion_rels = [r for r in relations if r.get("predicate") == "possible_occludes"]

    overlapping_pairs = []
    for pair in scene.get("pairwise_geometry") or []:
        ids = pair.get("layer_ids") or []
        if len(ids) == 2 and ids[0] in retained and ids[1] in retained and (pair.get("intersection_pixels") or 0) > 0:
            overlapping_pairs.append(ids)

    result: Dict[str, Dict[str, Any]] = {}

    def put(cat: str, eligible: bool, reason: str) -> None:
        result[cat] = {"eligible": eligible, "reason": reason}

    put(
        "direct_ask",
        n >= 1,
        f"{n} retained layer(s); needs at least 1 coherent recognizable object."
    )
    put(
        "attribute_binding",
        n >= 2 and len(directional) >= 1,
        f"{n} retained layer(s), {len(directional)} directional relation(s); needs a target, an anchor and a unique identifying relation."
    )
    put(
        "spatial",
        n >= 2 and len(directional) >= 1,
        f"{n} retained layer(s), {len(directional)} directional relation(s); needs 2+ layers with an unambiguous spatial relation."
    )
    put(
        "occlusion",
        len(occlusion_rels) >= 1 or len(overlapping_pairs) >= 1,
        f"{len(occlusion_rels)} possible-occlusion relation(s), {len(overlapping_pairs)} overlapping retained pair(s); needs meaningful overlap plus reliable front/behind ordering — z-index alone is insufficient."
    )
    put(
        "counting",
        n >= 2,
        f"{n} retained layer(s); needs separately represented objects forming a complete countable set."
    )
    put(
        "multi_hop_reference",
        n >= 3 and len(relations) >= 2,
        f"{n} retained layer(s), {len(relations)} relation(s); needs 3+ layers and a unique chain of 2+ relations."
    )
    return result


def suggest_categories(scene: Dict[str, Any], review: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Automatic suggestions with explanations. Human decisions stay authoritative."""
    eligibility = category_eligibility(scene, review)
    suggestions = []
    for cat in CATEGORIES:
        info = eligibility[cat]
        suggestions.append(
            {
                "category": cat,
                "label": CATEGORY_LABELS[cat],
                "suggested": info["eligible"],
                "eligible": True,
                "explanation": info["reason"],
            }
        )
    return suggestions


# ---------------------------------------------------------------------------
# Acceptance validation
# ---------------------------------------------------------------------------

def validate_acceptance(scene: Dict[str, Any], review: Dict[str, Any]) -> List[str]:
    """Return a list of blocking problems; empty list means acceptable."""
    problems: List[str] = []
    layers = scene.get("layers") or []

    if _is_blank(review.get("reviewer_id")):
        problems.append("Reviewer ID is missing.")

    retained: List[int] = []
    for layer in layers:
        lid = layer.get("layer_id")
        lr = _layer_review(review, lid)
        decision = lr.get("decision")
        if decision not in ("keep", "exclude"):
            problems.append(f"Layer {lid}: quality decision is missing.")
            continue
        if decision == "exclude":
            if _is_blank(lr.get("exclusion_reason")):
                problems.append(f"Layer {lid}: exclusion reason is required.")
            elif lr.get("exclusion_reason") not in EXCLUSION_REASONS:
                problems.append(f"Layer {lid}: unknown exclusion reason '{lr.get('exclusion_reason')}'.")
            elif lr.get("exclusion_reason") == "other" and _is_blank(lr.get("exclusion_note")):
                problems.append(f"Layer {lid}: exclusion reason 'other' needs a note.")
            continue
        retained.append(lid)
        if _is_blank(lr.get("label")):
            problems.append(f"Layer {lid}: canonical short label is required for retained layers.")
        if _is_blank(lr.get("caption")):
            problems.append(f"Layer {lid}: reviewed caption is required for retained layers.")
        if lr.get("geometry_status") not in ("correct", "minor_correction", "major_problem"):
            problems.append(f"Layer {lid}: geometry review status is required.")
        elif lr.get("geometry_status") == "major_problem":
            problems.append(
                f"Layer {lid}: geometry is marked as a major problem — correct it, exclude the layer, or reject the scene."
            )
        bbox = lr.get("bbox_override")
        if bbox is not None:
            ok = (
                isinstance(bbox, (list, tuple)) and len(bbox) == 4
                and all(isinstance(v, (int, float)) for v in bbox)
                and bbox[0] < bbox[2] and bbox[1] < bbox[3]
                and bbox[0] >= 0 and bbox[1] >= 0
                and bbox[2] <= (scene.get("width") or 0) and bbox[3] <= (scene.get("height") or 0)
            )
            if not ok:
                problems.append(f"Layer {lid}: corrected bbox must satisfy 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height.")

    if not retained:
        problems.append("At least one retained foreground layer is required to accept a scene.")

    if _is_blank(review.get("reviewed_image_caption")):
        problems.append("Reviewed BLIP-2 overall image caption is required.")
    if _is_blank(review.get("reviewed_image_caption_llava")):
        problems.append("Reviewed LLaVA overall image caption is required.")
    if _is_blank(review.get("reviewed_background_caption")):
        problems.append("Reviewed BLIP-2 background caption is required.")
    if _is_blank(review.get("reviewed_background_caption_llava")):
        problems.append("Reviewed LLaVA background caption is required.")
    questions = reviewed_questions(review)
    if not questions:
        problems.append("At least one question is required for a selected category.")
    else:
        answers = reviewed_answers(review, questions)
        for category in questions:
            if category not in answers:
                problems.append(f"An answer is required for the {CATEGORY_LABELS[category]} question.")

    primary = review.get("primary_category")
    suitable = review.get("suitable_categories") or []
    if _is_blank(primary):
        problems.append("Exactly one primary question category is required.")
    elif primary not in CATEGORIES:
        problems.append(f"Unknown primary category '{primary}'.")
    else:
        if primary not in suitable:
            problems.append("The primary category must also be marked as suitable.")
    for cat in suitable:
        if cat not in CATEGORIES:
            problems.append(f"Unknown suitable category '{cat}'.")

    quality = review.get("overall_quality")
    if quality is not None and (not isinstance(quality, int) or not 1 <= quality <= 5):
        problems.append("Overall quality rating must be an integer from 1 to 5.")

    return problems


def validate_rejection(review: Dict[str, Any]) -> List[str]:
    problems: List[str] = []
    if _is_blank(review.get("reviewer_id")):
        problems.append("Reviewer ID is missing.")
    return problems


# ---------------------------------------------------------------------------
# Curated record creation
# ---------------------------------------------------------------------------

def build_curated_scene(scene: Dict[str, Any], review: Dict[str, Any], missing_report: Dict[str, Any]) -> Dict[str, Any]:
    """Create the accepted curated scene. Assumes validate_acceptance passed."""
    retained = set(retained_layer_ids(scene, review))
    background = scene.get("background") or {}
    questions = reviewed_questions(review)
    answers = reviewed_answers(review, questions)

    curated_layers = []
    excluded_layers = []
    for layer in scene.get("layers") or []:
        lid = layer["layer_id"]
        lr = _layer_review(review, lid)
        if lid not in retained:
            excluded_layers.append(
                {
                    "layer_id": lid,
                    "z_index": layer.get("z_index"),
                    "path": layer.get("path"),
                    "exclusion_reason": lr.get("exclusion_reason"),
                    "exclusion_note": lr.get("exclusion_note") or None,
                }
            )
            continue
        curated = {
            "layer_id": lid,
            "native_instance_key": layer.get("native_instance_key"),
            "role": layer.get("role"),
            "z_index": layer.get("z_index"),
            "path": layer.get("path"),
            "label": lr["label"].strip(),
            "label_source": "human_review",
            "caption": lr["caption"].strip(),
            "caption_quality": "human_verified",
            "layer_quality": "good",
            "mask": layer.get("mask"),
            "geometry": layer.get("geometry"),
            "geometry_review": {
                "status": lr.get("geometry_status"),
                "bbox_override_xyxy": lr.get("bbox_override"),
                "centroid_override_xy": lr.get("centroid_override"),
                "note": lr.get("geometry_note") or None,
            },
        }
        curated_layers.append(curated)

    relations = _retained_relations(scene, retained, review)
    pairwise = [
        pair
        for pair in scene.get("pairwise_geometry") or []
        if len(pair.get("layer_ids") or []) == 2
        and pair["layer_ids"][0] in retained
        and pair["layer_ids"][1] in retained
    ]

    original_layer_provenance = [
        {
            "layer_id": layer.get("layer_id"),
            "label": layer.get("label"),
            "label_source": layer.get("label_source"),
            "captions": layer.get("captions"),
            "caption_quality": layer.get("caption_quality"),
        }
        for layer in scene.get("layers") or []
    ]

    record = {
        "scene_id": scene["scene_id"],
        "subset": scene.get("subset"),
        "render_subset": scene.get("render_subset"),
        "image_id": scene.get("image_id"),
        "width": scene.get("width"),
        "height": scene.get("height"),
        "full_path": scene.get("full_path"),
        "render_dir": scene.get("render_dir"),
        "image_ref": scene.get("image_ref"),
        "captions": {
            "image": review["reviewed_image_caption"].strip(),
            "background": review["reviewed_background_caption"].strip(),
            "image_by_model": {
                "blip2": review["reviewed_image_caption"].strip(),
                "llava": review["reviewed_image_caption_llava"].strip(),
            },
            "background_by_model": {
                "blip2": review["reviewed_background_caption"].strip(),
                "llava": review["reviewed_background_caption_llava"].strip(),
            },
        },
        "background": {
            "layer_id": background.get("layer_id", 0),
            "role": background.get("role", "background"),
            "z_index": background.get("z_index", 0),
            "path": background.get("path"),
            "caption": review["reviewed_background_caption"].strip(),
            "captions": {
                "blip2": review["reviewed_background_caption"].strip(),
                "llava": review["reviewed_background_caption_llava"].strip(),
            },
            "caption_source": "human_review",
        },
        "layers": curated_layers,
        "excluded_layers": excluded_layers,
        "relations": relations,
        "pairwise_geometry": pairwise,
        "categories": {
            "primary_category": review["primary_category"],
            "additional_suitable_categories": [
                c for c in (review.get("suitable_categories") or []) if c != review["primary_category"]
            ],
        },
        "questions": [
            {"category": category, "question": questions[category], "answer": answers[category]}
            for category in (review.get("suitable_categories") or [])
            if category in questions
        ],
        "review": {
            "reviewer_id": review["reviewer_id"],
            "decision": "accepted",
            "overall_quality": review.get("overall_quality"),
            "human_questions": questions,
            "human_answers": answers,
            "duration_seconds": review.get("duration_seconds"),
            "reviewed_at": review.get("finalized_at"),
            "missing_field_report": {
                "technical_nulls_preserved": missing_report.get("technical_ok", []),
            },
        },
        "provenance": {
            "original_captions": scene.get("captions"),
            "original_background_captions": background.get("captions"),
            "original_layers": original_layer_provenance,
            "original_relations": scene.get("relations"),
            "annotation": scene.get("annotation"),
            "inventory": scene.get("inventory"),
            "validation": scene.get("validation"),
            "captioning_models": scene.get("captioning_models"),
            "categories_image_level": scene.get("categories"),
            "category_scope": scene.get("category_scope"),
        },
    }
    return record


REQUIRED_CURATED_SCENE_FIELDS = (
    "scene_id",
    "subset",
    "image_id",
    "full_path",
    ("captions", "image"),
    ("captions", "background"),
    ("captions", "image_by_model", "blip2"),
    ("captions", "image_by_model", "llava"),
    ("captions", "background_by_model", "blip2"),
    ("captions", "background_by_model", "llava"),
    ("categories", "primary_category"),
    ("review", "reviewer_id"),
)
REQUIRED_CURATED_LAYER_FIELDS = ("label", "label_source", "caption", "path")


def validate_curated_scene(record: Dict[str, Any]) -> List[str]:
    """Final guard: the exported curated record must contain no null/empty
    required values. Runs at acceptance time and again during export."""
    problems: List[str] = []

    def get_path(obj: Any, key_path: Tuple[str, ...]) -> Any:
        for key in key_path:
            if not isinstance(obj, dict):
                return None
            obj = obj.get(key)
        return obj

    for field in REQUIRED_CURATED_SCENE_FIELDS:
        key_path = field if isinstance(field, tuple) else (field,)
        if _is_blank(get_path(record, key_path)):
            problems.append(f"Curated scene field {'.'.join(key_path)} is null/empty.")

    layers = record.get("layers")
    if not layers:
        problems.append("Curated scene has no retained layers.")
    for layer in layers or []:
        for field in REQUIRED_CURATED_LAYER_FIELDS:
            if _is_blank(layer.get(field)):
                problems.append(f"Curated layer {layer.get('layer_id')} field {field} is null/empty.")
        if layer.get("label_source") != "human_review":
            problems.append(f"Curated layer {layer.get('layer_id')} label_source must be 'human_review'.")

    primary = get_path(record, ("categories", "primary_category"))
    if primary not in CATEGORIES:
        problems.append(f"Curated scene primary_category '{primary}' is not a known category.")

    questions = record.get("questions")
    if not isinstance(questions, list) or not questions:
        problems.append("Curated scene must contain at least one categorized human question.")
    else:
        for question in questions:
            if (
                not isinstance(question, dict)
                or question.get("category") not in CATEGORIES
                or _is_blank(question.get("question"))
                or _is_blank(question.get("answer"))
            ):
                problems.append("Each curated human question needs a known category, question text, and answer.")

    for rel in record.get("relations") or []:
        layer_ids = {l.get("layer_id") for l in record.get("layers") or []}
        if rel.get("subject_layer_id") not in layer_ids or rel.get("object_layer_id") not in layer_ids:
            problems.append("Curated scene contains a relation touching an excluded layer.")

    return problems


def build_rejected_record(scene: Dict[str, Any], review: Dict[str, Any], missing_report: Dict[str, Any]) -> Dict[str, Any]:
    layer_assessments = []
    for layer in scene.get("layers") or []:
        lr = _layer_review(review, layer.get("layer_id"))
        layer_assessments.append(
            {
                "layer_id": layer.get("layer_id"),
                "decision": lr.get("decision"),
                "exclusion_reason": lr.get("exclusion_reason"),
            }
        )
    return {
        "scene_id": scene["scene_id"],
        "subset": scene.get("subset"),
        "image_id": scene.get("image_id"),
        "full_path": scene.get("full_path"),
        "decision": "rejected",
        "layer_assessments": layer_assessments,
        "missing_field_findings": {
            "human_required_unresolved": missing_report.get("human_required", []),
            "blocking": missing_report.get("blocking", []),
        },
        "reviewer_id": review.get("reviewer_id"),
        "reviewed_at": review.get("finalized_at"),
        "duration_seconds": review.get("duration_seconds"),
    }
