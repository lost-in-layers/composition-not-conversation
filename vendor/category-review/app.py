"""Local, single-reviewer category-only UI for the selected MuLAn scenes."""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parents[1]
HUMAN_REVIEW = REPO_ROOT / "vendor" / "human-review"
CACHE_DIR = HUMAN_REVIEW / "runtime" / "cache"
INDEX_PATH = CACHE_DIR / "scene_index.json"
RUNTIME_DIR = Path(os.environ.get("CATEGORY_REVIEW_RUNTIME", APP_ROOT / "runtime")).resolve()
QUEUE_PATH = RUNTIME_DIR / "queue.json"
STATE_PATH = RUNTIME_DIR / "state.json"
EXPORT_PATH = Path(
    os.environ.get(
        "CATEGORY_REVIEW_EXPORT",
        REPO_ROOT / "data" / "derived" / "mulan-category-review" / "category_selections.json",
    )
).resolve()

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
QUEUE_SEED = 20260719
QUESTION_TARGET = 350
STATE_SCHEMA = "mulan-category-review-v1"

app = FastAPI(title="MuLAn Category Review", docs_url=None, redoc_url=None)
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
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_index() -> dict[str, Any]:
    index = read_json(INDEX_PATH, None)
    if not isinstance(index, dict) or not isinstance(index.get("scenes"), dict):
        raise HTTPException(
            status_code=503,
            detail="The human-review scene cache is not ready. Start the existing human-review app and build its index once.",
        )
    return index


def interleave(left: list[str], right: list[str]) -> list[str]:
    result: list[str] = []
    for index in range(max(len(left), len(right))):
        if index < len(left):
            result.append(left[index])
        if index < len(right):
            result.append(right[index])
    return result


def load_queue(index: dict[str, Any]) -> list[str]:
    existing = read_json(QUEUE_PATH, None)
    known = set(index["scenes"])
    if isinstance(existing, list) and existing and len(existing) == len(known) and set(existing) == known:
        return existing
    lvis = list((index.get("subsets", {}).get("lvis") or {}).get("scene_ids") or [])
    laion = list((index.get("subsets", {}).get("laion") or {}).get("scene_ids") or [])
    rng = random.Random(QUEUE_SEED)
    rng.shuffle(lvis)
    rng.shuffle(laion)
    queue = interleave(lvis, laion)
    atomic_json(QUEUE_PATH, queue)
    return queue


def blank_state() -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA,
        "reviewer_id": "category_reviewer_1",
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
    state.setdefault("drafts", {})
    state.setdefault("decisions", {})
    state.setdefault("deferred", [])
    state.setdefault("history", [])
    state["cursor"] = min(max(int(state.get("cursor", 0)), 0), max(0, len(queue) - 1))
    return state


def fetch_scene(index: dict[str, Any], scene_id: str) -> dict[str, Any]:
    entry = index["scenes"].get(scene_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Unknown scene: {scene_id}")
    cache_path = CACHE_DIR / entry["file"]
    with cache_path.open("rb") as handle:
        handle.seek(int(entry["offset"]))
        raw = handle.read(int(entry["length"]))
    scene = json.loads(raw.decode("utf-8"))
    if scene.get("scene_id") != scene_id:
        raise HTTPException(status_code=503, detail="The shared scene cache is stale. Rebuild it in the existing human-review app.")
    return scene


def file_checks(scene: dict[str, Any]) -> dict[str, bool]:
    def exists(relative: str | None) -> bool:
        return bool(relative and (REPO_ROOT / relative).is_file())

    checks = {
        "full": exists(scene.get("full_path")),
        "background": exists((scene.get("background") or {}).get("path")),
        "progressive_strip": exists(f"{scene.get('render_dir', '')}/progressive_strip.png"),
    }
    for layer in scene.get("layers") or []:
        checks[f"layer_{layer.get('layer_id')}"] = exists(layer.get("path"))
    return checks


def category_suggestions(scene: dict[str, Any]) -> list[dict[str, Any]]:
    layers = scene.get("layers") or []
    relations = scene.get("relations") or []
    directional = [r for r in relations if r.get("predicate") in {"left_of", "right_of", "above", "below", "near", "contains"}]
    overlaps = [p for p in scene.get("pairwise_geometry") or [] if (p.get("intersection_pixels") or 0) > 0]
    flags = {
        "direct_ask": len(layers) >= 1,
        "attribute_binding": len(layers) >= 2 and bool(directional),
        "spatial": len(layers) >= 2 and bool(directional),
        "occlusion": any(r.get("predicate") == "possible_occludes" for r in relations) or bool(overlaps),
        "counting": len(layers) >= 2,
        "multi_hop_reference": len(layers) >= 3 and len(relations) >= 2,
    }
    return [{"category": category, "suggested": flags[category]} for category in CATEGORIES]


def validate_categories(categories: list[str], primary: str | None, *, accepting: bool) -> tuple[list[str], str | None]:
    unique = list(dict.fromkeys(categories))
    if any(category not in CATEGORIES for category in unique):
        raise HTTPException(status_code=422, detail="Unknown category")
    if primary is not None and primary not in CATEGORIES:
        raise HTTPException(status_code=422, detail="Unknown primary category")
    if primary is not None and primary not in unique:
        unique.append(primary)
    if accepting and not unique:
        raise HTTPException(status_code=422, detail="Select at least one category before accepting.")
    if accepting and primary not in unique:
        raise HTTPException(status_code=422, detail="Choose one primary category before accepting.")
    return unique, primary


def status_for(state: dict[str, Any], scene_id: str) -> str:
    decision = state["decisions"].get(scene_id)
    if decision:
        return decision["decision"]
    if scene_id in state["deferred"]:
        return "deferred"
    if scene_id in state["drafts"]:
        return "draft"
    return "unseen"


def progress(index: dict[str, Any], queue: list[str], state: dict[str, Any]) -> dict[str, Any]:
    decisions = state["decisions"]
    accepted = [value for value in decisions.values() if value.get("decision") == "accepted"]
    rejected = [value for value in decisions.values() if value.get("decision") == "rejected"]
    counts = {category: 0 for category in CATEGORIES}
    primary_counts = {category: 0 for category in CATEGORIES}
    for decision in accepted:
        for category in decision.get("categories") or []:
            if category in counts:
                counts[category] += 1
        primary = decision.get("primary_category")
        if primary in primary_counts:
            primary_counts[primary] += 1
    deferred = set(state["deferred"])
    drafts = [scene_id for scene_id in state["drafts"] if scene_id not in decisions and scene_id not in deferred]
    expected = sum(int(value) for value in (index.get("selection", {}).get("expected") or {}).values())
    return {
        "available": len(queue),
        "selected_total": expected or len(queue),
        "unavailable": max(0, (expected or len(queue)) - len(queue)),
        "reviewed": len(accepted) + len(rejected),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "deferred": len(deferred),
        "drafts": len(drafts),
        "question_slots": sum(len(item.get("categories") or []) for item in accepted),
        "question_target": QUESTION_TARGET,
        "category_counts": counts,
        "primary_counts": primary_counts,
    }


def next_open(queue: list[str], state: dict[str, Any], start: int) -> int | None:
    decisions = state["decisions"]
    deferred = set(state["deferred"])
    for offset in range(1, len(queue) + 1):
        position = (start + offset) % len(queue)
        scene_id = queue[position]
        if scene_id not in decisions and scene_id not in deferred:
            return position
    for offset in range(1, len(queue) + 1):
        position = (start + offset) % len(queue)
        if queue[position] not in decisions:
            return position
    return None


def export_records(index: dict[str, Any], queue: list[str], state: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for scene_id in queue:
        decision = state["decisions"].get(scene_id)
        if not decision or decision.get("decision") != "accepted":
            continue
        output.append(
            {
                "image_id": index["scenes"][scene_id].get("image_id"),
                "categories": {
                    "primary": decision.get("primary_category"),
                    "selected": decision.get("categories") or [],
                },
            }
        )
    return output


def write_export(index: dict[str, Any], queue: list[str], state: dict[str, Any]) -> list[dict[str, Any]]:
    records = export_records(index, queue, state)
    atomic_json(EXPORT_PATH, records)
    return records


def payload(scene_id: str) -> dict[str, Any]:
    with lock:
        index = load_index()
        queue = load_queue(index)
        state = load_state(queue)
        if scene_id not in index["scenes"]:
            raise HTTPException(status_code=404, detail=f"Unknown scene: {scene_id}")
        scene = fetch_scene(index, scene_id)
        return {
            "scene": scene,
            "draft": state["drafts"].get(scene_id),
            "decision": state["decisions"].get(scene_id),
            "status": status_for(state, scene_id),
            "file_checks": file_checks(scene),
            "suggestions": category_suggestions(scene),
            "queue_position": queue.index(scene_id),
            "queue_total": len(queue),
            "progress": progress(index, queue, state),
        }


class CategoryBody(BaseModel):
    categories: list[str] = Field(default_factory=list)
    primary_category: str | None = None


class FinalizeBody(CategoryBody):
    decision: str


class NavigateBody(BaseModel):
    direction: str | None = None
    scene_id: str | None = None


@app.get("/api/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/api/bootstrap")
def bootstrap() -> dict[str, Any]:
    with lock:
        index = load_index()
        queue = load_queue(index)
        state = load_state(queue)
        if not queue:
            raise HTTPException(status_code=503, detail="No cached selected scenes are available.")
        scene_id = queue[state["cursor"]]
        return {"scene_id": scene_id, "progress": progress(index, queue, state)}


@app.get("/api/scene/{scene_id}")
def get_scene(scene_id: str) -> dict[str, Any]:
    return payload(scene_id)


@app.post("/api/scene/{scene_id}/draft")
def save_draft(scene_id: str, body: CategoryBody) -> dict[str, Any]:
    with lock:
        index = load_index()
        queue = load_queue(index)
        state = load_state(queue)
        if scene_id in state["decisions"]:
            raise HTTPException(status_code=409, detail="This scene is already finalized.")
        categories, primary = validate_categories(body.categories, body.primary_category, accepting=False)
        if categories or primary:
            state["drafts"][scene_id] = {"categories": categories, "primary_category": primary}
        else:
            state["drafts"].pop(scene_id, None)
        atomic_json(STATE_PATH, state)
        return {"saved": True, "progress": progress(index, queue, state)}


@app.post("/api/scene/{scene_id}/finalize")
def finalize(scene_id: str, body: FinalizeBody) -> dict[str, Any]:
    if body.decision not in {"accepted", "rejected"}:
        raise HTTPException(status_code=422, detail="Decision must be accepted or rejected.")
    with lock:
        index = load_index()
        queue = load_queue(index)
        state = load_state(queue)
        if scene_id not in index["scenes"]:
            raise HTTPException(status_code=404, detail=f"Unknown scene: {scene_id}")
        categories, primary = validate_categories(body.categories, body.primary_category, accepting=body.decision == "accepted")
        previous = state["decisions"].get(scene_id)
        state["history"].append({"scene_id": scene_id, "previous": previous, "draft": state["drafts"].get(scene_id)})
        state["decisions"][scene_id] = {
            "decision": body.decision,
            "image_id": index["scenes"][scene_id].get("image_id"),
            "categories": categories if body.decision == "accepted" else [],
            "primary_category": primary if body.decision == "accepted" else None,
        }
        state["drafts"].pop(scene_id, None)
        state["deferred"] = [value for value in state["deferred"] if value != scene_id]
        current = queue.index(scene_id)
        following = next_open(queue, state, current)
        if following is not None:
            state["cursor"] = following
        atomic_json(STATE_PATH, state)
        write_export(index, queue, state)
        next_scene_id = queue[state["cursor"]] if following is not None else None
        return {"ok": True, "next_scene_id": next_scene_id, "progress": progress(index, queue, state)}


@app.post("/api/scene/{scene_id}/defer")
def defer(scene_id: str) -> dict[str, Any]:
    with lock:
        index = load_index()
        queue = load_queue(index)
        state = load_state(queue)
        if scene_id not in state["decisions"] and scene_id not in state["deferred"]:
            state["deferred"].append(scene_id)
        current = queue.index(scene_id)
        following = next_open(queue, state, current)
        if following is not None:
            state["cursor"] = following
        atomic_json(STATE_PATH, state)
        return {
            "next_scene_id": queue[state["cursor"]] if following is not None else None,
            "progress": progress(index, queue, state),
        }


@app.post("/api/navigate")
def navigate(body: NavigateBody) -> dict[str, Any]:
    with lock:
        index = load_index()
        queue = load_queue(index)
        state = load_state(queue)
        if body.scene_id:
            if body.scene_id not in queue:
                raise HTTPException(status_code=404, detail="Scene ID is not in the category-review queue.")
            state["cursor"] = queue.index(body.scene_id)
        elif body.direction == "previous":
            state["cursor"] = (state["cursor"] - 1) % len(queue)
        elif body.direction == "next":
            state["cursor"] = (state["cursor"] + 1) % len(queue)
        elif body.direction == "deferred":
            pending = [scene_id for scene_id in state["deferred"] if scene_id not in state["decisions"]]
            if not pending:
                raise HTTPException(status_code=404, detail="There are no deferred scenes.")
            state["cursor"] = queue.index(pending[0])
        else:
            raise HTTPException(status_code=422, detail="Choose previous, next, deferred, or a scene ID.")
        atomic_json(STATE_PATH, state)
        return {"scene_id": queue[state["cursor"]]}


@app.post("/api/export")
def export_now() -> dict[str, Any]:
    with lock:
        index = load_index()
        queue = load_queue(index)
        state = load_state(queue)
        records = write_export(index, queue, state)
        try:
            display_path = EXPORT_PATH.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            display_path = str(EXPORT_PATH)
        return {"records": len(records), "path": display_path}


@app.get("/api/export/download")
def download_export() -> FileResponse:
    export_now()
    return FileResponse(EXPORT_PATH, filename="category_selections.json", media_type="application/json")


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
    if candidate.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"} or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(candidate)


app.mount("/", StaticFiles(directory=APP_ROOT / "static", html=True), name="static")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8792)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
