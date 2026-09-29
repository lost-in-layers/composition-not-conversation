"""Classification of missing / null / empty values in source scenes.

Every null or empty value observed in a scene is placed into one of three
buckets:

``human_required``
    Values the reviewer must supply before the scene can be accepted
    (canonical layer labels, reviewed captions, unverified caption quality).

``technical_ok``
    Optional provenance/technical fields that may legitimately stay null
    (native checksum matches when no native checksum existed, empty inventory
    error strings, empty image-level category candidate lists). These are
    preserved verbatim and never "completed" by a human.

``blocking``
    Missing source values that cannot be fixed by typing a value in a form:
    a missing image/layer file or an unusable mask. These require layer
    exclusion, scene rejection, or deferral.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .paths import RepoPaths

# Technical/provenance fields (relative to their parent object) that may be
# null or empty without any human action.
TECHNICAL_NULL_OK_SCENE = {
    "annotation.actual_sha256",
    "annotation.actual_size_bytes",
    "annotation.matches_inventory_sha256",
    "annotation.sha256",
    "annotation.size_bytes",
    "inventory.error",
    "categories",
    "image_ref",
}
TECHNICAL_NULL_OK_LAYER = {
    "mask.native_alpha_sha256",
    "mask.matches_native_alpha",
    "native_instance_key",
}


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def classify_scene(
    scene: Dict[str, Any],
    review: Optional[Dict[str, Any]] = None,
    paths: Optional[RepoPaths] = None,
    check_files: bool = False,
) -> Dict[str, Any]:
    """Classify missing values for one scene, optionally overlaying the
    current review draft so completed fields stop being reported as missing.
    """
    review = review or {}
    layer_reviews: Dict[str, Any] = review.get("layer_reviews") or {}

    human_required: List[Dict[str, Any]] = []
    technical_ok: List[Dict[str, Any]] = []
    blocking: List[Dict[str, Any]] = []

    def add(bucket: List[Dict[str, Any]], field: str, detail: str, layer_id: Optional[int] = None) -> None:
        item: Dict[str, Any] = {"field": field, "detail": detail}
        if layer_id is not None:
            item["layer_id"] = layer_id
        bucket.append(item)

    # ---- scene-level captions -------------------------------------------
    captions = scene.get("captions") or {}
    if all(_is_blank(v) for v in captions.values()) or not captions:
        add(blocking, "captions", "No source caption for the full image; reviewer must write one from scratch or reject.")
    if _is_blank(review.get("reviewed_image_caption")):
        add(human_required, "reviewed_image_caption", "Reviewed BLIP-2 overall image caption is required before acceptance.")
    if _is_blank(review.get("reviewed_image_caption_llava")):
        add(human_required, "reviewed_image_caption_llava", "Reviewed LLaVA overall image caption is required before acceptance.")

    background = scene.get("background") or {}
    bg_captions = background.get("captions") or {}
    if all(_is_blank(v) for v in bg_captions.values()) or not bg_captions:
        add(blocking, "background.captions", "No source caption for the background layer.")
    if _is_blank(review.get("reviewed_background_caption")):
        add(human_required, "reviewed_background_caption", "Reviewed BLIP-2 background caption is required before acceptance.")
    if _is_blank(review.get("reviewed_background_caption_llava")):
        add(human_required, "reviewed_background_caption_llava", "Reviewed LLaVA background caption is required before acceptance.")

    # ---- scene-level technical nulls ------------------------------------
    annotation = scene.get("annotation") or {}
    for key in ("actual_sha256", "actual_size_bytes", "matches_inventory_sha256", "sha256", "size_bytes"):
        if key in annotation and annotation.get(key) is None:
            add(technical_ok, f"annotation.{key}", "No native checksum/size available; legitimately null.")
    inventory = scene.get("inventory") or {}
    if inventory.get("error", None) == "":
        add(technical_ok, "inventory.error", "Empty inventory error string; nothing to complete.")
    if scene.get("categories") == []:
        add(technical_ok, "categories", "No image-level category candidates for this subset; legitimately empty.")

    # ---- file existence (optional, needs repo paths) --------------------
    if check_files and paths is not None:
        for field, rel in (("full_path", scene.get("full_path")), ("background.path", background.get("path"))):
            if _is_blank(rel):
                add(blocking, field, "Source path missing from scene record.")
            elif not (paths.repo_root / rel).exists():
                add(blocking, field, f"Image file not found on disk: {rel}")

    # ---- per-layer -------------------------------------------------------
    for layer in scene.get("layers") or []:
        layer_id = layer.get("layer_id")
        lr = layer_reviews.get(str(layer_id)) or layer_reviews.get(layer_id) or {}
        excluded = lr.get("decision") == "exclude"

        if layer.get("label") is None:
            if excluded:
                add(technical_ok, "label", "Layer excluded by reviewer; no canonical label needed.", layer_id)
            elif _is_blank(lr.get("label")):
                add(human_required, "label", "Canonical short label must be assigned by the reviewer.", layer_id)
        if layer.get("label_source") is None and not excluded and _is_blank(lr.get("label")):
            add(human_required, "label_source", 'Will be set to "human_review" once a canonical label is accepted.', layer_id)

        source_caption = (layer.get("captions") or {})
        if all(_is_blank(v) for v in source_caption.values()) or not source_caption:
            if excluded:
                add(technical_ok, "captions", "Layer excluded; missing source caption is moot.", layer_id)
            else:
                add(blocking, "captions", "Layer has no source caption at all; reviewer must write one or exclude the layer.", layer_id)
        if not excluded and _is_blank(lr.get("caption")):
            add(human_required, "caption", "Reviewed layer caption is required for every retained layer.", layer_id)

        if layer.get("caption_quality") == "unverified" and _is_blank(lr.get("caption")) and not excluded:
            add(human_required, "caption_quality", "Layer caption has not been human-verified yet.", layer_id)

        mask = layer.get("mask") or {}
        for key in ("native_alpha_sha256", "matches_native_alpha"):
            if key in mask and mask.get(key) is None:
                add(technical_ok, f"mask.{key}", "No native alpha checksum was available; legitimately null.", layer_id)
        if layer.get("native_instance_key") is None:
            add(technical_ok, "native_instance_key", "No native instance key recorded; legitimately null.", layer_id)

        if _is_blank(layer.get("path")):
            add(blocking, "path", "Layer has no image path; exclude the layer or reject the scene.", layer_id)
        elif check_files and paths is not None and not (paths.repo_root / layer["path"]).exists():
            add(blocking, "path", f"Layer image not found on disk: {layer['path']}", layer_id)

        if not excluded and lr.get("decision") is None:
            add(human_required, "layer_decision", "Layer quality decision (keep / exclude) is required.", layer_id)
        if not excluded and lr.get("decision") == "keep" and _is_blank(lr.get("geometry_status")):
            add(human_required, "geometry_status", "Geometry review status is required for every retained layer.", layer_id)

    # ---- decision-level requirements ------------------------------------
    if _is_blank(review.get("primary_category")):
        add(human_required, "primary_category", "Exactly one primary question category must be chosen.")
    if review.get("category_decision_made") is not True and not review.get("suitable_categories"):
        add(human_required, "category_suitability", "Explicit category-suitability decision is required.")
    suitable = review.get("suitable_categories") or []
    questions = review.get("human_questions") or {}
    filled_questions = {
        category: value.strip()
        for category, value in questions.items()
        if category in suitable and isinstance(value, str) and value.strip()
    }
    has_question = bool(filled_questions)
    if not has_question:
        legacy = review.get("human_question")
        primary = review.get("primary_category")
        if primary in suitable and isinstance(legacy, str) and legacy.strip():
            filled_questions[primary] = legacy.strip()
            has_question = True
    if not has_question:
        add(human_required, "human_questions", "At least one question is required for a category selected in Step 4.")
    else:
        answers = review.get("human_answers") or {}
        for category in filled_questions:
            answer = answers.get(category)
            if not isinstance(answer, str) or not answer.strip():
                add(human_required, "human_answers", f"Answer required for the {category} question.")

    per_layer_counts: Dict[str, Dict[str, int]] = {}
    for bucket_name, bucket in (("human_required", human_required), ("technical_ok", technical_ok), ("blocking", blocking)):
        for item in bucket:
            lid = item.get("layer_id")
            if lid is None:
                continue
            slot = per_layer_counts.setdefault(str(lid), {"human_required": 0, "technical_ok": 0, "blocking": 0})
            slot[bucket_name] += 1

    return {
        "human_required": human_required,
        "technical_ok": technical_ok,
        "blocking": blocking,
        "counts": {
            "human_required": len(human_required),
            "technical_ok": len(technical_ok),
            "blocking": len(blocking),
        },
        "per_layer": per_layer_counts,
    }
