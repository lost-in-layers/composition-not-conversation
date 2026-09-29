"""Generate question shards and layer-evidence annotations for Layered-VQA.

Each question is processed by three GPT-5.6 Luna calls by default:

1. question-only semantic sharding (no image and no gold answer),
2. multimodal layer-evidence annotation, and
3. integrated multimodal verification/correction.

The JSONL output is an append-only checkpoint. A separate enriched JSON copy
places the generated fields directly on each selected question for human review.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import io
import json
import os
import random
import re
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path("data/derived/mulan-human-review/reviewer_2/accepted_scenes.json")
DEFAULT_OUTPUT = Path(
    "data/derived/mulan-human-review/reviewer_2/question_shards.jsonl"
)
DEFAULT_ENRICHED_OUTPUT = Path(
    "data/derived/mulan-human-review/reviewer_2/"
    "accepted_scenes_with_shards_and_evidence.json"
)
DEFAULT_PROMPT_DIR = Path("configs/question_sharding")
CATEGORIES = (
    "direct_ask",
    "attribute_binding",
    "spatial",
    "occlusion",
    "counting",
    "multi_hop_reference",
)
PROMPT_VERSION = "layered-vqa-shards-and-evidence-v3"
SCHEMA_VERSION = "layered-vqa-shards-and-evidence-v3"


EvidenceRole = Literal[
    "answer_target",
    "attribute_source",
    "reference_entity",
    "reference_landmark",
    "relational_bridge",
    "occluded_target",
    "occluder",
    "count_member",
    "spatial_context",
    "background_context",
    "text_source",
]


class AtomicUnit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit_id: int = Field(ge=1, le=10)
    source_span: str = Field(min_length=1)
    function: Literal[
        "intent",
        "answer_type",
        "target",
        "attribute",
        "relation",
        "reference",
        "scope",
        "condition",
    ]


class QuestionShard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shard_id: int = Field(ge=1, le=6)
    role: Literal[
        "intent",
        "answer_type",
        "target",
        "attribute",
        "relation",
        "reference",
        "scope",
        "condition",
    ]
    text: str = Field(min_length=1)
    source_unit_ids: list[int] = Field(min_length=1, max_length=10)


class ShardPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    atomic_units: list[AtomicUnit] = Field(min_length=3, max_length=10)
    shards: list[QuestionShard] = Field(min_length=3, max_length=6)
    information_preserved: Literal["yes"]
    added_visual_facts: Literal["no"]


class EvidenceDependency(BaseModel):
    model_config = ConfigDict(extra="forbid")

    before: int = Field(ge=0)
    after: int = Field(ge=0)
    reason: Literal[
        "reference_resolution",
        "relation_resolution",
        "attribute_binding",
        "occlusion_resolution",
        "count_scope_resolution",
        "answer_extraction",
    ]


class LayerRoleAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layer_id: int = Field(ge=0)
    role: EvidenceRole


class QuestionShardSupportAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shard_id: int = Field(ge=1, le=6)
    layer_ids: list[int] = Field(max_length=8)


class EvidencePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gold_layer_ids: list[int] = Field(min_length=1, max_length=8)
    layer_roles: dict[str, EvidenceRole]
    minimal_sufficient_layer_sets: list[list[int]] = Field(min_length=1, max_length=8)
    question_shard_support: dict[str, list[int]]
    evidence_dependencies: list[EvidenceDependency] = Field(max_length=16)
    distractor_layer_ids: list[int] = Field(max_length=8)
    evidence_notes: str = Field(min_length=1, max_length=1200)
    confidence: Literal["high", "medium", "low"]


class EvidenceStructuredPlan(BaseModel):
    """Strict-schema API shape; normalized to EvidencePlan before saving."""

    model_config = ConfigDict(extra="forbid")

    gold_layer_ids: list[int] = Field(min_length=1, max_length=8)
    layer_roles: list[LayerRoleAssignment] = Field(min_length=1, max_length=8)
    minimal_sufficient_layer_sets: list[list[int]] = Field(min_length=1, max_length=8)
    question_shard_support: list[QuestionShardSupportAssignment] = Field(
        min_length=3, max_length=6
    )
    evidence_dependencies: list[EvidenceDependency] = Field(max_length=16)
    distractor_layer_ids: list[int] = Field(max_length=8)
    evidence_notes: str = Field(min_length=1, max_length=1200)
    confidence: Literal["high", "medium", "low"]


class IntegratedVerifiedPlan(EvidenceStructuredPlan):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["pass", "revised", "flag_for_human"]
    issues: list[
        Literal[
            "information_loss",
            "added_question_information",
            "answer_leakage",
            "unclear_initial_intent",
            "forced_question_padding",
            "wrong_evidence_layer",
            "missing_evidence_layer",
            "non_minimal_evidence_set",
            "wrong_layer_role",
            "wrong_shard_layer_mapping",
            "wrong_dependency_direction",
            "wrong_distractor_partition",
            "ambiguous_visual_evidence",
            "caption_pixel_conflict",
        ]
    ]
    atomic_units: list[AtomicUnit] = Field(min_length=3, max_length=10)
    shards: list[QuestionShard] = Field(min_length=3, max_length=6)


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def question_id(scene_id: str, category: str) -> str:
    return f"{scene_id}::{category}"


def public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def simplify_scene(scene: dict[str, Any]) -> dict[str, Any]:
    background = scene["background"]
    layers = scene.get("layers", [])
    active_ids = {0, *(int(layer["layer_id"]) for layer in layers)}
    relations = []
    for relation in scene.get("relations", []):
        subject = relation.get("subject_layer_id")
        object_id = relation.get("object_layer_id")
        if subject in active_ids and object_id in active_ids:
            relations.append(
                {
                    "subject_layer_id": subject,
                    "predicate": relation.get("predicate"),
                    "object_layer_id": object_id,
                    "score": relation.get("score"),
                }
            )

    return {
        "scene_id": scene["scene_id"],
        "full_path": scene["full_path"],
        "width": scene.get("width"),
        "height": scene.get("height"),
        "candidate_layer_ids": sorted(active_ids),
        "background": {
            "layer_id": 0,
            "path": background["path"],
            "role": "background",
            "caption": background.get("caption", ""),
        },
        "layers": [
            {
                "layer_id": int(layer["layer_id"]),
                "path": layer["path"],
                "z_index": layer.get("z_index"),
                "label": layer.get("label"),
                "caption": layer.get("caption", ""),
                "bbox_normalized": (layer.get("geometry") or {}).get("bbox_normalized"),
                "centroid_normalized": (layer.get("geometry") or {}).get(
                    "support_centroid_normalized"
                ),
            }
            for layer in layers
        ],
        "relations": relations,
    }


def load_questions(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    scenes = payload.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("Input must contain a top-level 'scenes' list.")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for scene in scenes:
        scene_id = scene.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError("Every scene must have a non-empty scene_id.")
        simplified_scene = simplify_scene(scene)
        for question in scene.get("questions", []):
            category = question.get("category")
            text = question.get("question")
            answer = question.get("answer")
            if category not in CATEGORIES:
                raise ValueError(
                    f"Unsupported category {category!r} in scene {scene_id}."
                )
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty question in scene {scene_id}.")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"Empty answer for {scene_id} category {category}.")

            record_id = question_id(scene_id, category)
            if record_id in seen:
                raise ValueError(f"Duplicate question id: {record_id}")
            seen.add(record_id)
            records.append(
                {
                    "question_id": record_id,
                    "scene_id": scene_id,
                    "category": category,
                    "question": text.strip(),
                    "answer": answer.strip(),
                    "_scene": simplified_scene,
                }
            )

    declared_count = payload.get("count")
    if declared_count is not None and declared_count != len(scenes):
        raise ValueError(
            f"Input declares {declared_count} scenes but contains {len(scenes)}."
        )
    return payload, records


def read_prompt_bundle(prompt_dir: Path, category: str) -> dict[str, str]:
    paths = {
        "base": prompt_dir / "base.txt",
        "category": prompt_dir / f"{category}.txt",
        "evidence_base": prompt_dir / "evidence_base.txt",
        "evidence_category": prompt_dir / f"evidence_{category}.txt",
        "integrated_verification": prompt_dir / "integrated_verification.txt",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing prompt files: " + ", ".join(missing))
    return {
        name: path.read_text(encoding="utf-8").strip() for name, path in paths.items()
    }


def prompt_digest(bundle: dict[str, str]) -> str:
    return sha256_text(
        PROMPT_VERSION + "\n" + "\n".join(bundle[key] for key in sorted(bundle))
    )


def usage_dict(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return dict(usage)


def response_metadata(response: Any) -> dict[str, Any]:
    return {"response_id": response.id, "usage": usage_dict(response)}


def structured_to_evidence(
    plan: EvidenceStructuredPlan | IntegratedVerifiedPlan,
) -> EvidencePlan:
    role_ids = [assignment.layer_id for assignment in plan.layer_roles]
    if len(role_ids) != len(set(role_ids)):
        raise ValueError("Duplicate layer_id in layer_roles assignments.")
    shard_ids = [assignment.shard_id for assignment in plan.question_shard_support]
    if len(shard_ids) != len(set(shard_ids)):
        raise ValueError("Duplicate shard_id in question_shard_support assignments.")
    return EvidencePlan(
        gold_layer_ids=plan.gold_layer_ids,
        layer_roles={
            str(assignment.layer_id): assignment.role for assignment in plan.layer_roles
        },
        minimal_sufficient_layer_sets=plan.minimal_sufficient_layer_sets,
        question_shard_support={
            str(assignment.shard_id): assignment.layer_ids
            for assignment in plan.question_shard_support
        },
        evidence_dependencies=plan.evidence_dependencies,
        distractor_layer_ids=plan.distractor_layer_ids,
        evidence_notes=plan.evidence_notes,
        confidence=plan.confidence,
    )


def call_sharder(
    client: OpenAI,
    record: dict[str, Any],
    bundle: dict[str, str],
    args: argparse.Namespace,
) -> tuple[ShardPlan, dict[str, Any]]:
    response = client.responses.parse(
        model=args.model,
        reasoning={"effort": args.reasoning_effort},
        instructions=f"{bundle['base']}\n\n{bundle['category']}",
        input=(
            f"CATEGORY: {record['category']}\nCANONICAL QUESTION:\n{record['question']}"
        ),
        text_format=ShardPlan,
        max_output_tokens=6000,
        store=False,
        timeout=args.timeout,
    )
    if response.output_parsed is None:
        raise RuntimeError("Question sharding returned no parsed output.")
    return response.output_parsed, response_metadata(response)


def checkerboard(size: tuple[int, int], cell: int = 24) -> Image.Image:
    canvas = Image.new("RGB", size, (224, 224, 224))
    draw = ImageDraw.Draw(canvas)
    for y in range(0, size[1], cell):
        for x in range(0, size[0], cell):
            if (x // cell + y // cell) % 2:
                draw.rectangle(
                    (x, y, min(x + cell - 1, size[0]), min(y + cell - 1, size[1])),
                    fill=(176, 176, 176),
                )
    return canvas


@lru_cache(maxsize=256)
def image_data_url(path_text: str, isolate_rgba: bool) -> str:
    path = Path(path_text)
    with Image.open(path) as source:
        if isolate_rgba:
            rgba = source.convert("RGBA")
            visible = checkerboard(rgba.size).convert("RGBA")
            image = Image.alpha_composite(visible, rgba).convert("RGB")
        else:
            image = source.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def scene_manifest(scene: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene_id": scene["scene_id"],
        "candidate_layer_ids": scene["candidate_layer_ids"],
        "background": {
            key: scene["background"].get(key) for key in ("layer_id", "role", "caption")
        },
        "layers": [
            {
                key: layer.get(key)
                for key in (
                    "layer_id",
                    "z_index",
                    "label",
                    "caption",
                    "bbox_normalized",
                    "centroid_normalized",
                )
            }
            for layer in scene["layers"]
        ],
        "auxiliary_geometry_relations": scene["relations"],
    }


def build_visual_content(
    record: dict[str, Any],
    leading_text: str,
    image_detail: Literal["low", "high", "auto"],
) -> list[dict[str, Any]]:
    scene = record["_scene"]
    content: list[dict[str, Any]] = [
        {"type": "input_text", "text": leading_text},
        {
            "type": "input_text",
            "text": (
                "FULL COMPOSITE IMAGE. This provides global context; choose "
                "evidence IDs only from the numbered layers that follow."
            ),
        },
        {
            "type": "input_image",
            "image_url": image_data_url(
                str(resolve_repo_path(scene["full_path"])), False
            ),
            "detail": image_detail,
        },
        {
            "type": "input_text",
            "text": (
                "LAYER 0 (background). Reviewed caption: "
                f"{scene['background']['caption']}"
            ),
        },
        {
            "type": "input_image",
            "image_url": image_data_url(
                str(resolve_repo_path(scene["background"]["path"])), False
            ),
            "detail": image_detail,
        },
    ]
    for layer in scene["layers"]:
        content.extend(
            [
                {
                    "type": "input_text",
                    "text": (
                        f"LAYER {layer['layer_id']} (isolated RGBA inspection "
                        "card on a synthetic gray checkerboard; the "
                        "checkerboard is not scene evidence). Reviewed caption: "
                        f"{layer['caption']}\nAuxiliary label: {layer['label']}"
                    ),
                },
                {
                    "type": "input_image",
                    "image_url": image_data_url(
                        str(resolve_repo_path(layer["path"])), True
                    ),
                    "detail": image_detail,
                },
            ]
        )
    return content


def call_evidence_annotator(
    client: OpenAI,
    record: dict[str, Any],
    shard_plan: ShardPlan,
    bundle: dict[str, str],
    args: argparse.Namespace,
) -> tuple[EvidencePlan, dict[str, Any]]:
    leading_text = (
        f"CATEGORY: {record['category']}\n"
        f"QUESTION: {record['question']}\n"
        f"HUMAN GOLD ANSWER: {record['answer']}\n\n"
        "QUESTION SHARDS:\n"
        f"{json.dumps([s.model_dump() for s in shard_plan.shards], indent=2)}\n\n"
        "SCENE MANIFEST:\n"
        f"{json.dumps(scene_manifest(record['_scene']), ensure_ascii=False, indent=2)}"
    )
    response = client.responses.parse(
        model=args.model,
        reasoning={"effort": args.reasoning_effort},
        instructions=(f"{bundle['evidence_base']}\n\n{bundle['evidence_category']}"),
        input=[
            {
                "role": "user",
                "content": build_visual_content(
                    record, leading_text, args.image_detail
                ),
            }
        ],
        text_format=EvidenceStructuredPlan,
        max_output_tokens=8000,
        store=False,
        timeout=args.timeout,
    )
    if response.output_parsed is None:
        raise RuntimeError("Evidence annotation returned no parsed output.")
    return structured_to_evidence(response.output_parsed), response_metadata(response)


def call_integrated_verifier(
    client: OpenAI,
    record: dict[str, Any],
    atomic_units: list[AtomicUnit],
    shards: list[QuestionShard],
    evidence_plan: EvidencePlan,
    bundle: dict[str, str],
    args: argparse.Namespace,
    deterministic_feedback: list[str] | None = None,
) -> tuple[IntegratedVerifiedPlan, dict[str, Any]]:
    proposal = {
        "atomic_units": [unit.model_dump() for unit in atomic_units],
        "shards": [shard.model_dump() for shard in shards],
        **evidence_plan.model_dump(),
    }
    feedback_text = ""
    if deterministic_feedback:
        feedback_text = "\n\nDETERMINISTIC VALIDATOR FEEDBACK TO REPAIR:\n" + "\n".join(
            f"- {issue}" for issue in deterministic_feedback
        )
    leading_text = (
        f"CATEGORY: {record['category']}\n"
        f"QUESTION: {record['question']}\n"
        f"HUMAN GOLD ANSWER: {record['answer']}\n\n"
        "PROPOSED INTEGRATED ANNOTATION:\n"
        f"{json.dumps(proposal, ensure_ascii=False, indent=2)}\n\n"
        "SCENE MANIFEST:\n"
        f"{json.dumps(scene_manifest(record['_scene']), ensure_ascii=False, indent=2)}"
        f"{feedback_text}"
    )
    response = client.responses.parse(
        model=args.model,
        reasoning={"effort": args.reasoning_effort},
        instructions=(
            f"{bundle['integrated_verification']}\n\n"
            "QUESTION CATEGORY RUBRIC:\n"
            f"{bundle['category']}\n\n"
            "EVIDENCE CATEGORY RUBRIC:\n"
            f"{bundle['evidence_category']}"
        ),
        input=[
            {
                "role": "user",
                "content": build_visual_content(
                    record, leading_text, args.image_detail
                ),
            }
        ],
        text_format=IntegratedVerifiedPlan,
        max_output_tokens=12000,
        store=False,
        timeout=args.timeout,
    )
    if response.output_parsed is None:
        raise RuntimeError("Integrated verification returned no parsed output.")
    return response.output_parsed, response_metadata(response)


def validate_shards(
    question: str,
    answer: str,
    atomic_units: list[AtomicUnit],
    shards: list[QuestionShard],
) -> list[str]:
    issues: list[str] = []
    unit_ids = [unit.unit_id for unit in atomic_units]
    shard_ids = [shard.shard_id for shard in shards]
    if unit_ids != list(range(1, len(unit_ids) + 1)):
        issues.append("atomic_unit_ids_not_sequential")
    if shard_ids != list(range(1, len(shard_ids) + 1)):
        issues.append("shard_ids_not_sequential")
    if not shards or shards[0].role != "intent":
        issues.append("first_shard_is_not_intent")

    normalized_question = normalize_text(question)
    for unit in atomic_units:
        normalized_span = normalize_text(unit.source_span)
        if not normalized_span or normalized_span not in normalized_question:
            issues.append(f"source_span_not_in_question:{unit.unit_id}")

    valid_unit_ids = set(unit_ids)
    covered_unit_ids: set[int] = set()
    normalized_shards: list[str] = []
    for shard in shards:
        referenced = set(shard.source_unit_ids)
        if not referenced <= valid_unit_ids:
            issues.append(f"unknown_source_unit:{shard.shard_id}")
        covered_unit_ids.update(referenced & valid_unit_ids)
        normalized_shards.append(normalize_text(shard.text))
    if covered_unit_ids != valid_unit_ids:
        missing = sorted(valid_unit_ids - covered_unit_ids)
        issues.append("uncovered_atomic_units:" + ",".join(map(str, missing)))
    if len(set(normalized_shards)) != len(normalized_shards):
        issues.append("duplicate_shard_text")

    normalized_answer = normalize_text(answer)
    joined_shards = " ".join(normalized_shards)
    if (
        normalized_answer
        and normalized_answer not in normalized_question
        and normalized_answer in joined_shards
    ):
        issues.append("gold_answer_leakage")
    return list(dict.fromkeys(issues))


def parse_int_keys(mapping: dict[str, Any], field: str) -> tuple[set[int], list[str]]:
    parsed: set[int] = set()
    issues: list[str] = []
    for key in mapping:
        try:
            parsed.add(int(key))
        except ValueError:
            issues.append(f"{field}_has_non_integer_key:{key}")
    return parsed, issues


def validate_evidence(
    candidate_layer_ids: list[int],
    shards: list[QuestionShard],
    plan: EvidencePlan | IntegratedVerifiedPlan,
) -> list[str]:
    issues: list[str] = []
    candidates = set(candidate_layer_ids)
    gold = plan.gold_layer_ids
    if len(gold) != len(set(gold)):
        issues.append("duplicate_gold_layer_ids")
    if not set(gold) <= candidates:
        issues.append("gold_contains_unknown_layer")

    sufficient_sets = [set(values) for values in plan.minimal_sufficient_layer_sets]
    if any(not values for values in sufficient_sets):
        issues.append("empty_minimal_sufficient_set")
    if any(not values <= candidates for values in sufficient_sets):
        issues.append("sufficient_set_contains_unknown_layer")
    if sufficient_sets and sufficient_sets[0] != set(gold):
        issues.append("gold_is_not_first_sufficient_set")
    normalized_sets = [tuple(sorted(values)) for values in sufficient_sets]
    if len(normalized_sets) != len(set(normalized_sets)):
        issues.append("duplicate_minimal_sufficient_set")
    for index, values in enumerate(sufficient_sets):
        if any(other < values for other in sufficient_sets):
            issues.append(f"non_minimal_sufficient_set:{index}")

    evidence_union = set().union(*sufficient_sets) if sufficient_sets else set()
    role_ids, role_issues = parse_int_keys(plan.layer_roles, "layer_roles")
    issues.extend(role_issues)
    if role_ids != evidence_union:
        issues.append("layer_roles_do_not_match_evidence_union")

    shard_ids = {str(shard.shard_id) for shard in shards}
    support_ids, support_key_issues = parse_int_keys(
        plan.question_shard_support, "question_shard_support"
    )
    issues.extend(support_key_issues)
    if {str(value) for value in support_ids} != shard_ids:
        issues.append("question_shard_support_keys_incomplete")
    support_union: set[int] = set()
    for layer_ids in plan.question_shard_support.values():
        support = set(layer_ids)
        if not support <= candidates:
            issues.append("shard_support_contains_unknown_layer")
        if not support <= evidence_union:
            issues.append("shard_support_contains_distractor")
        support_union.update(support)
    if support_union != evidence_union:
        issues.append("shard_support_does_not_cover_evidence_union")

    distractors = set(plan.distractor_layer_ids)
    if len(plan.distractor_layer_ids) != len(distractors):
        issues.append("duplicate_distractor_layer_ids")
    if distractors != candidates - evidence_union:
        issues.append("distractors_do_not_partition_candidates")

    for dependency in plan.evidence_dependencies:
        if dependency.before == dependency.after:
            issues.append("self_evidence_dependency")
        if dependency.before not in evidence_union:
            issues.append("dependency_before_not_evidence")
        if dependency.after not in evidence_union:
            issues.append("dependency_after_not_evidence")
    return list(dict.fromkeys(issues))


def shardability_label(question: str, shard_count: int) -> str:
    word_count = len(re.findall(r"\S+", question))
    if shard_count == 3:
        return "structured_3_short" if word_count <= 10 else "natural_3"
    return "natural_4_plus"


def integrated_to_evidence(plan: IntegratedVerifiedPlan) -> EvidencePlan:
    return structured_to_evidence(plan)


def generate_one(
    client: OpenAI,
    record: dict[str, Any],
    bundle: dict[str, str],
    args: argparse.Namespace,
    input_sha256: str,
) -> dict[str, Any]:
    result_base = {
        "schema_version": SCHEMA_VERSION,
        **public_record(record),
    }
    try:
        shard_plan, shard_meta = call_sharder(client, record, bundle, args)
        evidence_plan, evidence_meta = call_evidence_annotator(
            client, record, shard_plan, bundle, args
        )

        verifier_meta: dict[str, Any] | None = None
        verifier_metas: list[dict[str, Any]] = []
        verifier_verdict = "not_run"
        verifier_issues: list[str] = []
        atomic_units = shard_plan.atomic_units
        shards = shard_plan.shards
        final_evidence = evidence_plan
        confidence = evidence_plan.confidence
        if not args.skip_verification:
            verified, verifier_meta = call_integrated_verifier(
                client,
                record,
                atomic_units,
                shards,
                evidence_plan,
                bundle,
                args,
            )
            verifier_metas.append(verifier_meta)
            verifier_verdict = verified.verdict
            verifier_issues = list(verified.issues)
            atomic_units = verified.atomic_units
            shards = verified.shards
            final_evidence = integrated_to_evidence(verified)
            confidence = verified.confidence

        deterministic_issues = validate_shards(
            record["question"], record["answer"], atomic_units, shards
        )
        deterministic_issues.extend(
            validate_evidence(
                record["_scene"]["candidate_layer_ids"],
                shards,
                final_evidence,
            )
        )
        deterministic_issues = list(dict.fromkeys(deterministic_issues))

        if deterministic_issues and not args.skip_verification:
            corrected, verifier_meta = call_integrated_verifier(
                client,
                record,
                atomic_units,
                shards,
                final_evidence,
                bundle,
                args,
                deterministic_feedback=deterministic_issues,
            )
            verifier_metas.append(verifier_meta)
            verifier_verdict = corrected.verdict
            verifier_issues = list(dict.fromkeys([*verifier_issues, *corrected.issues]))
            atomic_units = corrected.atomic_units
            shards = corrected.shards
            final_evidence = integrated_to_evidence(corrected)
            confidence = corrected.confidence
            deterministic_issues = validate_shards(
                record["question"], record["answer"], atomic_units, shards
            )
            deterministic_issues.extend(
                validate_evidence(
                    record["_scene"]["candidate_layer_ids"],
                    shards,
                    final_evidence,
                )
            )
            deterministic_issues = list(dict.fromkeys(deterministic_issues))

        flagged = verifier_verdict == "flag_for_human" or bool(deterministic_issues)

        return {
            **result_base,
            "question_word_count": len(re.findall(r"\S+", record["question"])),
            "shard_count": len(shards),
            "shardability": shardability_label(record["question"], len(shards)),
            "atomic_units": [unit.model_dump() for unit in atomic_units],
            "shards": [shard.model_dump() for shard in shards],
            **final_evidence.model_dump(),
            "validation": {
                "status": "flagged" if flagged else "pass",
                "model_verdict": verifier_verdict,
                "model_issues": verifier_issues,
                "deterministic_issues": deterministic_issues,
                "human_review_status": "pending",
                "confidence": confidence,
            },
            "generation": {
                "status": "ok",
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "image_detail": args.image_detail,
                "verification_enabled": not args.skip_verification,
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": prompt_digest(bundle),
                "source_sha256": input_sha256,
                "sharding_response": shard_meta,
                "evidence_response": evidence_meta,
                "verification_response": verifier_meta,
                "verification_responses": verifier_metas,
            },
        }
    except Exception as exc:
        return {
            **result_base,
            "generation": {
                "status": "error",
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "image_detail": args.image_detail,
                "verification_enabled": not args.skip_verification,
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": prompt_digest(bundle),
                "source_sha256": input_sha256,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        }


def load_latest_results(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            record_id = record.get("question_id")
            if isinstance(record_id, str):
                latest[record_id] = record
    return latest


class JsonlWriter:
    def __init__(self, path: Path, overwrite: bool) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite:
            self.path.write_text("", encoding="utf-8")
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(stable_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def compatible_result(
    result: dict[str, Any],
    record: dict[str, Any],
    bundle: dict[str, str],
    args: argparse.Namespace,
    input_sha256: str,
) -> bool:
    generation = result.get("generation", {})
    return (
        result.get("schema_version") == SCHEMA_VERSION
        and generation.get("status") == "ok"
        and generation.get("model") == args.model
        and generation.get("reasoning_effort") == args.reasoning_effort
        and generation.get("image_detail") == args.image_detail
        and generation.get("verification_enabled") == (not args.skip_verification)
        and generation.get("prompt_sha256") == prompt_digest(bundle)
        and generation.get("source_sha256") == input_sha256
        and result.get("question") == record["question"]
        and result.get("answer") == record["answer"]
    )


def select_temp_records(
    records: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    if count >= len(records):
        return records
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["category"]].append(record)
    rng = random.Random(seed)
    for values in grouped.values():
        rng.shuffle(values)

    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for category in CATEGORIES:
            if grouped[category] and len(selected) < count:
                selected.append(grouped[category].pop())
                progressed = True
        if not progressed:
            break
    return selected


def default_temp_paths(count: int) -> tuple[Path, Path]:
    root = DEFAULT_OUTPUT.parent
    return (
        root / f"question_shards.temp-{count}.jsonl",
        root / f"accepted_scenes_with_shards_and_evidence.temp-{count}.json",
    )


ANNOTATION_KEYS = (
    "question_word_count",
    "shard_count",
    "shardability",
    "atomic_units",
    "gold_layer_ids",
    "layer_roles",
    "minimal_sufficient_layer_sets",
    "question_shard_support",
    "evidence_dependencies",
    "distractor_layer_ids",
    "evidence_notes",
    "validation",
    "generation",
)


def write_enriched_dataset(
    payload: dict[str, Any],
    selected_records: list[dict[str, Any]],
    latest: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    selected_ids = {record["question_id"] for record in selected_records}
    enriched = copy.deepcopy(payload)
    enriched["schema_version"] = SCHEMA_VERSION
    enriched["kind"] = "accepted_scenes_with_question_shards_and_layer_evidence"
    enriched["source_kind"] = payload.get("kind")
    enriched_scenes = []
    for scene in enriched["scenes"]:
        kept_questions = []
        for question in scene.get("questions", []):
            record_id = question_id(scene["scene_id"], question["category"])
            if record_id not in selected_ids:
                continue
            result = latest.get(record_id)
            if result and result.get("generation", {}).get("status") == "ok":
                question["question_shards"] = result["shards"]
                for key in ANNOTATION_KEYS:
                    if key in result:
                        question[key] = result[key]
            else:
                question["generation"] = (
                    result.get("generation", {}) if result else {"status": "missing"}
                )
            kept_questions.append(question)
        if kept_questions:
            scene["questions"] = kept_questions
            enriched_scenes.append(scene)
    enriched["scenes"] = enriched_scenes
    enriched["count"] = len(enriched_scenes)
    enriched["question_count"] = len(selected_records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(enriched, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)


def write_summary(
    output_path: Path,
    enriched_path: Path,
    source_path: Path,
    source_sha256: str,
    records: list[dict[str, Any]],
    latest: dict[str, dict[str, Any]],
) -> Path:
    selected_ids = {record["question_id"] for record in records}
    results = [latest[item] for item in selected_ids if item in latest]
    successes = [
        item for item in results if item.get("generation", {}).get("status") == "ok"
    ]
    failures = [
        item for item in results if item.get("generation", {}).get("status") == "error"
    ]
    summary = {
        "schema_version": f"{SCHEMA_VERSION}-summary",
        "source": str(source_path),
        "source_sha256": source_sha256,
        "checkpoint_output": str(output_path),
        "enriched_output": str(enriched_path),
        "selected": len(records),
        "successful": len(successes),
        "failed": len(failures),
        "flagged": sum(
            item.get("validation", {}).get("status") == "flagged" for item in successes
        ),
        "by_category": dict(
            sorted(Counter(item["category"] for item in successes).items())
        ),
        "by_shard_count": {
            str(key): value
            for key, value in sorted(
                Counter(item["shard_count"] for item in successes).items()
            )
        },
        "by_confidence": dict(
            sorted(
                Counter(
                    item.get("validation", {}).get("confidence", "unknown")
                    for item in successes
                ).items()
            )
        ),
        "prompt_version": PROMPT_VERSION,
    }
    summary_path = output_path.with_suffix(".summary.json")
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)
    return summary_path


def parse_categories(values: list[str] | None) -> set[str]:
    if not values:
        return set(CATEGORIES)
    selected: set[str] = set()
    for value in values:
        selected.update(part.strip() for part in value.split(",") if part.strip())
    unknown = selected - set(CATEGORIES)
    if unknown:
        raise ValueError("Unknown categories: " + ", ".join(sorted(unknown)))
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate question-only shards, minimal layer evidence, layer roles, "
            "and many-to-many shard/layer mappings with GPT-5.6 Luna."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", help="Checkpoint JSONL output path.")
    parser.add_argument(
        "--enriched-output",
        help="Accepted-scenes JSON copy enriched for human review.",
    )
    parser.add_argument("--prompt-dir", default=str(DEFAULT_PROMPT_DIR))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"))
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default=os.getenv("OPENAI_REASONING_EFFORT", "high"),
    )
    parser.add_argument(
        "--image-detail",
        choices=("low", "high", "auto"),
        default="high",
    )
    parser.add_argument(
        "--category",
        action="append",
        help="Repeat or pass comma-separated question categories.",
    )
    parser.add_argument(
        "--scene-id",
        action="append",
        help="Limit generation to one or more exact scene IDs.",
    )
    parser.add_argument(
        "--temp",
        type=int,
        default=0,
        metavar="X",
        help=(
            "Generate a deterministic category-balanced temporary sample of X "
            "questions. Without --temp, generate every filtered question."
        ),
    )
    parser.add_argument("--temp-seed", type=int, default=20260813)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip the recommended third integrated verification call.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the checkpoint instead of resuming compatible records.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs/prompts without API calls or output writes.",
    )
    return parser


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = build_parser().parse_args()
    try:
        if args.workers < 1:
            raise ValueError("--workers must be at least 1.")
        if args.temp < 0:
            raise ValueError("--temp cannot be negative.")

        input_path = resolve_repo_path(args.input)
        prompt_dir = resolve_repo_path(args.prompt_dir)
        categories = parse_categories(args.category)
        payload, all_records = load_questions(input_path)
        records = [
            record
            for record in all_records
            if record["category"] in categories
            and (not args.scene_id or record["scene_id"] in set(args.scene_id))
        ]
        if not records:
            raise ValueError("No questions matched the selected filters.")
        if args.temp:
            records = select_temp_records(records, args.temp, args.temp_seed)

        if args.temp:
            default_output, default_enriched = default_temp_paths(args.temp)
        else:
            default_output, default_enriched = (
                DEFAULT_OUTPUT,
                DEFAULT_ENRICHED_OUTPUT,
            )
        output_path = resolve_repo_path(args.output or default_output)
        enriched_path = resolve_repo_path(args.enriched_output or default_enriched)
        bundles = {
            category: read_prompt_bundle(prompt_dir, category)
            for category in categories
        }
        input_sha256 = sha256_file(input_path)

        missing_visuals = []
        for record in records:
            scene = record["_scene"]
            paths = [scene["full_path"], scene["background"]["path"]]
            paths.extend(layer["path"] for layer in scene["layers"])
            missing_visuals.extend(
                str(resolve_repo_path(path))
                for path in paths
                if not resolve_repo_path(path).is_file()
            )
        if missing_visuals:
            raise FileNotFoundError(
                "Missing visual inputs: " + ", ".join(missing_visuals[:10])
            )

        print(f"Scenes in source: {len(payload['scenes'])}")
        print(f"Questions in source: {len(all_records)}")
        print(f"Questions selected: {len(records)}")
        print(f"Model: {args.model}")
        print(f"Reasoning effort: {args.reasoning_effort}")
        print(f"Image detail: {args.image_detail}")
        print(f"Integrated verification: {not args.skip_verification}")
        print(f"Checkpoint: {output_path}")
        print(f"Enriched review file: {enriched_path}")

        if args.dry_run:
            counts = Counter(record["category"] for record in records)
            print("Category counts: " + stable_json(dict(sorted(counts.items()))))
            print("Prompt digests:")
            for category in sorted(bundles):
                print(f"  {category}: {prompt_digest(bundles[category])}")
            print("Dry run complete; no API calls or files were written.")
            return

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Add it to the repository .env file."
            )

        existing = {} if args.overwrite else load_latest_results(output_path)
        completed = {
            record["question_id"]
            for record in records
            if record["question_id"] in existing
            and compatible_result(
                existing[record["question_id"]],
                record,
                bundles[record["category"]],
                args,
                input_sha256,
            )
        }
        pending = [
            record for record in records if record["question_id"] not in completed
        ]
        print(f"Already complete and compatible: {len(completed)}")
        print(f"Pending: {len(pending)}")

        writer = JsonlWriter(output_path, overwrite=args.overwrite)
        client = OpenAI(api_key=api_key, max_retries=5)
        failures = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    generate_one,
                    client,
                    record,
                    bundles[record["category"]],
                    args,
                    input_sha256,
                ): record
                for record in pending
            }
            with tqdm(
                total=len(pending),
                desc="Generate shards + evidence",
                unit="question",
                dynamic_ncols=True,
            ) as progress:
                for future in as_completed(futures):
                    result = future.result()
                    writer.append(result)
                    if result.get("generation", {}).get("status") == "error":
                        failures += 1
                        tqdm.write(
                            f"ERROR {result['question_id']}: "
                            f"{result['generation'].get('error')}"
                        )
                    progress.update(1)

        latest = load_latest_results(output_path)
        write_enriched_dataset(payload, records, latest, enriched_path)
        summary_path = write_summary(
            output_path,
            enriched_path,
            input_path,
            input_sha256,
            records,
            latest,
        )
        successful = sum(
            latest.get(record["question_id"], {}).get("generation", {}).get("status")
            == "ok"
            for record in records
        )
        print(f"Successful: {successful}/{len(records)}")
        print(f"New failures: {failures}")
        print(f"Checkpoint: {output_path}")
        print(f"Enriched review file: {enriched_path}")
        print(f"Summary: {summary_path}")
        if failures or successful != len(records):
            raise SystemExit(2)
    except KeyboardInterrupt:
        print("Cancelled; completed JSONL records are resumable.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
