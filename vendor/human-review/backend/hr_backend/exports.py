"""Atomic maintenance of the final export files, one pair per reviewer.

After every final decision that reviewer's two exports are rebuilt from their
recorded decisions and written atomically:

    data/derived/mulan-human-review/reviewer_<n>/accepted_scenes.json
    data/derived/mulan-human-review/reviewer_<n>/rejected_scenes.json

Source scene JSON and PNGs are never modified.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List

from . import reviewers
from .atomicio import atomic_write_json
from .curation import validate_curated_scene
from .paths import RepoPaths


def _upgrade_curated_for_export(curated: Dict[str, Any], review: Dict[str, Any]) -> Dict[str, Any]:
    """Return an export-ready copy of an older curated record.

    Records accepted while the previous backend process was still running may
    have answers in ``record.review`` but question-only entries in the embedded
    ``curated_scene`` snapshot. Reconstruct the question list from the
    authoritative saved review without altering the review state.
    """
    upgraded = deepcopy(curated)
    questions = review.get("human_questions") or {}
    answers = review.get("human_answers") or {}
    suitable = review.get("suitable_categories") or []
    pairs = []
    for category in suitable:
        question = questions.get(category)
        answer = answers.get(category)
        if (
            isinstance(question, str)
            and question.strip()
            and isinstance(answer, str)
            and answer.strip()
        ):
            pairs.append(
                {
                    "category": category,
                    "question": question.strip(),
                    "answer": answer.strip(),
                }
            )
    if pairs:
        upgraded["questions"] = pairs
        review_block = upgraded.setdefault("review", {})
        review_block["human_questions"] = {
            pair["category"]: pair["question"] for pair in pairs
        }
        review_block["human_answers"] = {
            pair["category"]: pair["answer"] for pair in pairs
        }
    return upgraded


def write_exports(paths: RepoPaths, state: Dict[str, Any], reviewer: str) -> Dict[str, Any]:
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    problems: List[str] = []

    for scene_id in state.get("decision_order") or []:
        record = (state.get("decisions") or {}).get(scene_id)
        if not record:
            continue
        if record.get("decision") == "accepted":
            curated = record.get("curated_scene")
            if curated is None:
                problems.append(f"{scene_id}: accepted decision has no curated scene; skipped.")
                continue
            curated = _upgrade_curated_for_export(curated, record.get("review") or {})
            issues = validate_curated_scene(curated)
            if issues:
                problems.append(f"{scene_id}: curated scene failed export validation: {'; '.join(issues)}")
                continue
            accepted.append(curated)
        elif record.get("decision") == "rejected":
            rejected.append(record.get("rejected_record") or {"scene_id": scene_id})

    now = datetime.now(timezone.utc).isoformat()
    reviewer_id = reviewers.REVIEWER_ID[reviewer]
    accepted_payload = {
        "schema_version": "1.0",
        "kind": "mulan_human_review_accepted",
        "reviewer": reviewer_id,
        "generated_at_utc": now,
        "count": len(accepted),
        "scenes": accepted,
    }
    rejected_payload = {
        "schema_version": "1.0",
        "kind": "mulan_human_review_rejected",
        "reviewer": reviewer_id,
        "generated_at_utc": now,
        "count": len(rejected),
        "scenes": rejected,
    }
    accepted_path = paths.accepted_export_for(reviewer)
    rejected_path = paths.rejected_export_for(reviewer)
    paths.export_dir_for(reviewer).mkdir(parents=True, exist_ok=True)
    atomic_write_json(accepted_path, accepted_payload)
    atomic_write_json(rejected_path, rejected_payload)
    return {
        "accepted": len(accepted),
        "rejected": len(rejected),
        "problems": problems,
        "accepted_path": str(accepted_path),
        "rejected_path": str(rejected_path),
    }
