"""Progress and dashboard statistics derived from the review state."""
from __future__ import annotations

from typing import Any, Dict

from .curation import CATEGORIES

PILOT_TARGET_PER_CATEGORY = 100


def compute_progress(state: Dict[str, Any], queue: Dict[str, Any]) -> Dict[str, Any]:
    decisions = state.get("decisions") or {}
    accepted = {sid: d for sid, d in decisions.items() if d.get("decision") == "accepted"}
    rejected = {sid: d for sid, d in decisions.items() if d.get("decision") == "rejected"}

    category_counts = {cat: 0 for cat in CATEGORIES}
    subset_counts = {"lvis": 0, "laion": 0}
    for record in accepted.values():
        curated = record.get("curated_scene") or {}
        primary = ((curated.get("categories") or {}).get("primary_category"))
        if primary in category_counts:
            category_counts[primary] += 1
        subset = curated.get("subset")
        if subset in subset_counts:
            subset_counts[subset] += 1

    # This reviewer's assignment: review every scene in their own queue.
    total_in_queue = len(queue.get("order") or [])
    finalized = len(decisions)
    return {
        # The goal is to review the whole assigned set, so the target is the
        # reviewer's own queue size (their fixed ~600 images).
        "target_total": total_in_queue,
        "target_per_category": PILOT_TARGET_PER_CATEGORY,
        "reviewed": finalized,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "deferred": len(state.get("deferred") or []),
        "drafts": len(state.get("drafts") or {}),
        "finalized": finalized,
        "remaining_to_target": max(0, total_in_queue - finalized),
        "queue_total": total_in_queue,
        "queue_cursor": state.get("cursor", 0),
        "category_counts": category_counts,
        "category_quota_full": [cat for cat, n in category_counts.items() if n >= PILOT_TARGET_PER_CATEGORY],
        "accepted_by_subset": subset_counts,
        "reviewer_id": state.get("reviewer_id"),
        "reviewer_slot": state.get("reviewer_slot"),
    }


def compute_dashboard(state: Dict[str, Any], queue: Dict[str, Any]) -> Dict[str, Any]:
    decisions = state.get("decisions") or {}
    accepted = [d for d in decisions.values() if d.get("decision") == "accepted"]
    rejected = [d for d in decisions.values() if d.get("decision") == "rejected"]

    quality_hist: Dict[str, int] = {str(i): 0 for i in range(1, 6)}
    exclusion_reasons: Dict[str, int] = {}
    rejection_reasons: Dict[str, int] = {}
    caption_edited = 0
    caption_total = 0
    labels_completed = 0
    labels_needed = 0
    geometry_corrected = 0
    geometry_layers = 0
    durations = []

    for record in rejected:
        reason = (record.get("rejected_record") or {}).get("rejection_reason") or record.get("review", {}).get("rejection_reason")
        if reason:
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    for record in accepted:
        curated = record.get("curated_scene") or {}
        review = record.get("review") or {}
        quality = (curated.get("review") or {}).get("overall_quality")
        if isinstance(quality, int) and 1 <= quality <= 5:
            quality_hist[str(quality)] += 1
        duration = (curated.get("review") or {}).get("duration_seconds")
        if isinstance(duration, (int, float)) and duration >= 0:
            durations.append(duration)

        for excluded in curated.get("excluded_layers") or []:
            reason = excluded.get("exclusion_reason") or "unspecified"
            exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1

        provenance_layers = {
            layer.get("layer_id"): layer
            for layer in (curated.get("provenance") or {}).get("original_layers") or []
        }
        for layer in curated.get("layers") or []:
            labels_needed += 1
            if layer.get("label"):
                labels_completed += 1
            geometry_layers += 1
            geo_review = layer.get("geometry_review") or {}
            if geo_review.get("status") == "minor_correction" or geo_review.get("bbox_override_xyxy") or geo_review.get("centroid_override_xy"):
                geometry_corrected += 1
            original = provenance_layers.get(layer.get("layer_id")) or {}
            original_caption = (original.get("captions") or {}).get("blip2") or ""
            caption_total += 1
            if (layer.get("caption") or "").strip() != original_caption.strip():
                caption_edited += 1

        provenance = curated.get("provenance") or {}
        original_image_caption = ((provenance.get("original_captions") or {}).get("blip2")) or ""
        caption_total += 1
        if (curated.get("captions") or {}).get("image", "").strip() != original_image_caption.strip():
            caption_edited += 1

    return {
        "quality_distribution": quality_hist,
        "exclusion_reasons": exclusion_reasons,
        "rejection_reasons": rejection_reasons,
        "caption_edit_rate": (caption_edited / caption_total) if caption_total else None,
        "missing_label_completion_rate": (labels_completed / labels_needed) if labels_needed else None,
        "geometry_correction_rate": (geometry_corrected / geometry_layers) if geometry_layers else None,
        "mean_review_duration_seconds": (sum(durations) / len(durations)) if durations else None,
    }
