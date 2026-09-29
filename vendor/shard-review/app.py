"""Local single-reviewer UI for question shards and layer evidence."""
from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parents[1]
REVIEWER_DIR = REPO_ROOT / "data" / "derived" / "mulan-human-review" / "reviewer_2"
SCENES_PATH = Path(os.environ.get("SHARD_REVIEW_SCENES", REVIEWER_DIR / "accepted_scenes.json")).resolve()
SHARDS_PATH = Path(os.environ.get("SHARD_REVIEW_SHARDS", REVIEWER_DIR / "question_shards.jsonl")).resolve()
RUNTIME_DIR = Path(os.environ.get("SHARD_REVIEW_RUNTIME", APP_ROOT / "runtime")).resolve()
STATE_PATH = RUNTIME_DIR / "state.json"
EXPORT_PATH = Path(
    os.environ.get(
        "SHARD_REVIEW_EXPORT",
        REVIEWER_DIR / "question-shards-reviewed.json",
    )
).resolve()

STATE_SCHEMA = "layered-vqa-shard-review-v1"
ANNOTATION_FIELDS = (
    "atomic_units",
    "shards",
    "gold_layer_ids",
    "layer_roles",
    "minimal_sufficient_layer_sets",
    "question_shard_support",
    "evidence_dependencies",
    "distractor_layer_ids",
    "evidence_notes",
    "confidence",
)

app = FastAPI(title="Layered-VQA Shard Review", docs_url=None, redoc_url=None)
lock = threading.RLock()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_sources() -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[str], dict[str, int]]:
    scenes_payload = read_json(SCENES_PATH, None)
    if not isinstance(scenes_payload, dict) or not isinstance(scenes_payload.get("scenes"), list):
        raise HTTPException(status_code=503, detail=f"Accepted scenes not found: {SCENES_PATH}")
    scenes = {scene["scene_id"]: scene for scene in scenes_payload["scenes"]}

    latest: dict[str, dict[str, Any]] = {}
    malformed = 0
    if not SHARDS_PATH.is_file():
        raise HTTPException(status_code=503, detail=f"Shard checkpoint not found: {SHARDS_PATH}")
    with SHARDS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(record.get("question_id"), str):
                latest[record["question_id"]] = record

    queue: list[str] = []
    for scene in scenes_payload["scenes"]:
        for question in scene.get("questions", []):
            qid = f"{scene['scene_id']}::{question['category']}"
            if qid not in latest:
                raise HTTPException(
                    status_code=503,
                    detail=f"Checkpoint is missing question: {qid}",
                )
            queue.append(qid)
    successful = sum(
        latest[qid].get("generation", {}).get("status") == "ok"
        for qid in queue
    )
    stats = {
        "source_questions": sum(len(scene.get("questions", [])) for scene in scenes.values()),
        "generated": successful,
        "failed": len(queue) - successful,
        "malformed_lines": malformed,
    }
    return scenes_payload, latest, queue, stats


def blank_state() -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA,
        "cursor": 0,
        "drafts": {},
        "decisions": {},
        "deferred": [],
        "history": [],
    }


def load_state(queue: list[str]) -> dict[str, Any]:
    state = read_json(STATE_PATH, None)
    if not isinstance(state, dict) or state.get("schema_version") != STATE_SCHEMA:
        state = blank_state()
        atomic_json(STATE_PATH, state)
    for key, default in (("drafts", {}), ("decisions", {}), ("deferred", []), ("history", [])):
        state.setdefault(key, default)
    state["cursor"] = min(max(int(state.get("cursor", 0)), 0), max(0, len(queue) - 1))
    return state


def editable_annotation(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("generation", {}).get("status") != "ok":
        raise HTTPException(
            status_code=503,
            detail=f"Question generation is not successful: {record['question_id']}",
        )
    return {key: copy.deepcopy(record.get(key)) for key in ANNOTATION_FIELDS}


def validate_annotation(annotation: dict[str, Any], scene: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in ANNOTATION_FIELDS if key not in annotation]
    if missing:
        raise HTTPException(status_code=422, detail="Missing fields: " + ", ".join(missing))
    shards = annotation.get("shards")
    if not isinstance(shards, list) or not 3 <= len(shards) <= 6:
        raise HTTPException(status_code=422, detail="shards must contain 3–6 entries")
    atomic_units = annotation.get("atomic_units")
    if not isinstance(atomic_units, list) or len(atomic_units) < 3:
        raise HTTPException(status_code=422, detail="atomic_units must contain at least 3 entries")
    unit_ids = [item.get("unit_id") for item in atomic_units]
    if unit_ids != list(range(1, len(unit_ids) + 1)):
        raise HTTPException(status_code=422, detail="atomic unit IDs must be consecutive from 1")
    expected_shards = list(range(1, len(shards) + 1))
    if [item.get("shard_id") for item in shards] != expected_shards:
        raise HTTPException(status_code=422, detail="shard_id values must be consecutive from 1")
    if any(not str(item.get("text", "")).strip() for item in shards):
        raise HTTPException(status_code=422, detail="Every shard needs non-empty text")
    covered_units = {
        int(unit_id)
        for shard in shards
        for unit_id in shard.get("source_unit_ids", [])
    }
    if covered_units != set(unit_ids):
        raise HTTPException(status_code=422, detail="Shard source_unit_ids must cover every atomic unit exactly as a set")
    candidate_ids = {0, *(int(layer["layer_id"]) for layer in scene.get("layers", []))}
    selected_ids: set[int] = set()
    for values in annotation.get("minimal_sufficient_layer_sets") or []:
        selected_ids.update(int(value) for value in values)
    for field in ("gold_layer_ids", "distractor_layer_ids"):
        values = annotation.get(field)
        if not isinstance(values, list) or any(int(value) not in candidate_ids for value in values):
            raise HTTPException(status_code=422, detail=f"{field} contains an invalid layer ID")
    if not annotation.get("minimal_sufficient_layer_sets"):
        raise HTTPException(status_code=422, detail="At least one minimal sufficient layer set is required")
    if set(annotation["gold_layer_ids"]) != set(annotation["minimal_sufficient_layer_sets"][0]):
        raise HTTPException(status_code=422, detail="gold_layer_ids must equal the first minimal sufficient layer set")
    if set(map(int, annotation.get("layer_roles", {}).keys())) != selected_ids:
        raise HTTPException(status_code=422, detail="layer_roles must cover the evidence-layer union exactly")
    support = annotation.get("question_shard_support") or {}
    if set(support) != {str(value) for value in expected_shards}:
        raise HTTPException(status_code=422, detail="question_shard_support must have one key per shard")
    if set(annotation["distractor_layer_ids"]) != candidate_ids - selected_ids:
        raise HTTPException(status_code=422, detail="distractor_layer_ids must be the exact evidence complement")
    return annotation


def status_for(state: dict[str, Any], question_id: str) -> str:
    if question_id in state["decisions"]:
        return state["decisions"][question_id]["decision"]
    if question_id in state["deferred"]:
        return "deferred"
    if question_id in state["drafts"]:
        return "draft"
    return "unseen"


def progress(queue: list[str], state: dict[str, Any], stats: dict[str, int]) -> dict[str, Any]:
    decisions = state["decisions"]
    accepted = sum(item.get("decision") == "accepted" for item in decisions.values())
    rejected = sum(item.get("decision") == "rejected" for item in decisions.values())
    return {
        **stats,
        "available": len(queue),
        "reviewed": accepted + rejected,
        "accepted": accepted,
        "rejected": rejected,
        "deferred": sum(qid not in decisions for qid in state["deferred"]),
        "drafts": sum(qid not in decisions for qid in state["drafts"]),
        "remaining": len(queue) - accepted - rejected,
    }


def next_open(queue: list[str], state: dict[str, Any], start: int) -> int | None:
    decisions = state["decisions"]
    deferred = set(state["deferred"])
    for offset in range(1, len(queue) + 1):
        position = (start + offset) % len(queue)
        qid = queue[position]
        if qid not in decisions and qid not in deferred:
            return position
    for offset in range(1, len(queue) + 1):
        position = (start + offset) % len(queue)
        if queue[position] not in decisions:
            return position
    return None


def file_checks(scene: dict[str, Any]) -> dict[str, bool]:
    def exists(relative: str | None) -> bool:
        return bool(relative and (REPO_ROOT / relative).is_file())
    checks = {
        "full": exists(scene.get("full_path")),
        "background": exists((scene.get("background") or {}).get("path")),
        "progressive_strip": exists(f"{scene.get('render_dir', '')}/progressive_strip.png"),
    }
    for layer in scene.get("layers", []):
        checks[f"layer_{layer['layer_id']}"] = exists(layer.get("path"))
    return checks


def export_reviewed(scenes_payload: dict[str, Any], latest: dict[str, dict[str, Any]], state: dict[str, Any]) -> dict[str, Any]:
    exported = copy.deepcopy(scenes_payload)
    accepted_count = 0
    reviewed_questions = []
    for scene in exported["scenes"]:
        kept = []
        for question in scene.get("questions", []):
            qid = f"{scene['scene_id']}::{question['category']}"
            decision = state["decisions"].get(qid)
            if not decision or decision.get("decision") != "accepted":
                continue
            source = latest[qid]
            question.update(copy.deepcopy(decision["annotation"]))
            question["answer"] = decision.get("answer", source["answer"])
            question["question_id"] = qid
            question["model_validation"] = source.get("validation")
            question["generation"] = source.get("generation")
            question["human_review"] = {"status": "accepted", "edited": decision.get("edited", False)}
            kept.append(question)
            accepted_count += 1
        if kept:
            scene["questions"] = kept
            reviewed_questions.append(scene)
    exported["scenes"] = reviewed_questions
    exported["count"] = len(reviewed_questions)
    exported["question_count"] = accepted_count
    exported["kind"] = "accepted_scenes_with_human_reviewed_shards_and_layer_evidence"
    exported["schema_version"] = STATE_SCHEMA
    atomic_json(EXPORT_PATH, exported)
    return exported


def payload(question_id: str) -> dict[str, Any]:
    with lock:
        scenes_payload, latest, queue, stats = load_sources()
        if question_id not in queue:
            raise HTTPException(status_code=404, detail=f"Unknown successful question: {question_id}")
        state = load_state(queue)
        record = latest[question_id]
        scene = next(scene for scene in scenes_payload["scenes"] if scene["scene_id"] == record["scene_id"])
        decision = state["decisions"].get(question_id)
        draft = state["drafts"].get(question_id)
        annotation = copy.deepcopy(
            (decision or draft or {}).get("annotation")
            or editable_annotation(record)
        )
        answer = (decision or draft or {}).get("answer", record["answer"])
        return {
            "scene": scene,
            "record": record,
            "annotation": annotation,
            "answer": answer,
            "decision": decision,
            "status": status_for(state, question_id),
            "file_checks": file_checks(scene),
            "queue_position": queue.index(question_id),
            "queue_total": len(queue),
            "progress": progress(queue, state, stats),
        }


class AnnotationBody(BaseModel):
    annotation: dict[str, Any]
    answer: str


class FinalizeBody(AnnotationBody):
    decision: str


class NavigateBody(BaseModel):
    direction: str | None = None
    question_id: str | None = None


@app.get("/api/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/api/bootstrap")
def bootstrap() -> dict[str, Any]:
    with lock:
        _, _, queue, stats = load_sources()
        if not queue:
            raise HTTPException(status_code=503, detail="No questions are available.")
        state = load_state(queue)
        return {"question_id": queue[state["cursor"]], "progress": progress(queue, state, stats)}


@app.get("/api/question/{question_id:path}")
def get_question(question_id: str) -> dict[str, Any]:
    return payload(question_id)


@app.post("/api/question/{question_id:path}/draft")
def save_draft(question_id: str, body: AnnotationBody) -> dict[str, Any]:
    with lock:
        _, _, queue, stats = load_sources()
        if question_id not in queue:
            raise HTTPException(status_code=404, detail="Unknown question")
        state = load_state(queue)
        if question_id in state["decisions"]:
            raise HTTPException(status_code=409, detail="This question is already finalized.")
        state["drafts"][question_id] = {
            "annotation": copy.deepcopy(body.annotation),
            "answer": body.answer,
        }
        atomic_json(STATE_PATH, state)
        return {"saved": True, "progress": progress(queue, state, stats)}


@app.post("/api/question/{question_id:path}/finalize")
def finalize(question_id: str, body: FinalizeBody) -> dict[str, Any]:
    if body.decision not in {"accepted", "rejected"}:
        raise HTTPException(status_code=422, detail="Decision must be accepted or rejected.")
    with lock:
        scenes_payload, latest, queue, stats = load_sources()
        if question_id not in queue:
            raise HTTPException(status_code=404, detail="Unknown question")
        state = load_state(queue)
        scene = next(scene for scene in scenes_payload["scenes"] if scene["scene_id"] == latest[question_id]["scene_id"])
        annotation = validate_annotation(body.annotation, scene)
        answer = body.answer.strip()
        if not answer:
            raise HTTPException(status_code=422, detail="Human answer cannot be empty.")
        original = editable_annotation(latest[question_id])
        state["history"].append({"question_id": question_id, "previous": state["decisions"].get(question_id)})
        state["decisions"][question_id] = {
            "decision": body.decision,
            "annotation": annotation,
            "answer": answer,
            "edited": annotation != original or answer != latest[question_id]["answer"],
        }
        state["drafts"].pop(question_id, None)
        state["deferred"] = [value for value in state["deferred"] if value != question_id]
        following = next_open(queue, state, queue.index(question_id))
        if following is not None:
            state["cursor"] = following
        atomic_json(STATE_PATH, state)
        export_reviewed(scenes_payload, latest, state)
        return {
            "ok": True,
            "next_question_id": queue[state["cursor"]] if following is not None else None,
            "progress": progress(queue, state, stats),
        }


@app.post("/api/question/{question_id:path}/defer")
def defer(question_id: str) -> dict[str, Any]:
    with lock:
        _, _, queue, stats = load_sources()
        state = load_state(queue)
        if question_id not in state["decisions"] and question_id not in state["deferred"]:
            state["deferred"].append(question_id)
        following = next_open(queue, state, queue.index(question_id))
        if following is not None:
            state["cursor"] = following
        atomic_json(STATE_PATH, state)
        return {"next_question_id": queue[state["cursor"]] if following is not None else None, "progress": progress(queue, state, stats)}


@app.post("/api/navigate")
def navigate(body: NavigateBody) -> dict[str, str]:
    with lock:
        _, latest, queue, _ = load_sources()
        state = load_state(queue)
        if body.question_id:
            target = body.question_id
            if target not in queue and "::" not in target:
                matches = [qid for qid in queue if qid.startswith(target + "::")]
                if len(matches) == 1:
                    target = matches[0]
            if target not in queue:
                raise HTTPException(status_code=404, detail="Question ID is not in the generated queue.")
            state["cursor"] = queue.index(target)
        elif body.direction == "previous":
            state["cursor"] = (state["cursor"] - 1) % len(queue)
        elif body.direction == "next":
            state["cursor"] = (state["cursor"] + 1) % len(queue)
        elif body.direction == "flagged":
            flagged = [
                qid
                for qid in queue
                if latest[qid].get("validation", {}).get("status") == "flagged"
                and qid not in state["decisions"]
            ]
            if not flagged:
                raise HTTPException(status_code=404, detail="No unreviewed flagged questions remain.")
            state["cursor"] = queue.index(flagged[0])
        elif body.direction == "deferred":
            pending = [qid for qid in state["deferred"] if qid not in state["decisions"]]
            if not pending:
                raise HTTPException(status_code=404, detail="No deferred questions remain.")
            state["cursor"] = queue.index(pending[0])
        else:
            raise HTTPException(status_code=422, detail="Choose previous, next, flagged, deferred, or a question ID.")
        atomic_json(STATE_PATH, state)
        return {"question_id": queue[state["cursor"]]}


@app.post("/api/export")
def export_now() -> dict[str, Any]:
    with lock:
        scenes_payload, latest, queue, _ = load_sources()
        state = load_state(queue)
        exported = export_reviewed(scenes_payload, latest, state)
        return {"questions": exported["question_count"], "path": str(EXPORT_PATH.relative_to(REPO_ROOT))}


@app.get("/api/export/download")
def download_export() -> FileResponse:
    export_now()
    return FileResponse(EXPORT_PATH, filename=EXPORT_PATH.name, media_type="application/json")


@app.get("/api/image/{relative_path:path}")
def image(relative_path: str) -> FileResponse:
    normalized = relative_path.replace("\\", "/").lstrip("/")
    if not normalized.startswith("data/"):
        raise HTTPException(status_code=403, detail="Only repository data images are servable.")
    candidate = (REPO_ROOT / normalized).resolve()
    data_root = (REPO_ROOT / "data").resolve()
    try:
        candidate.relative_to(data_root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Invalid image path.") from exc
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(candidate)


app.mount("/", StaticFiles(directory=APP_ROOT / "static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8793)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
