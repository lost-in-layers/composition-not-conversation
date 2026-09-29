"""Local multi-reviewer UI for validating the consensus judge against people.

Reviewers read a finished conversation and decide one thing: does the model's
final answer match the reference answer? The eleven judges' verdicts are never
sent to the page. A reviewer who could see them would be checking the judges
rather than grading the conversation.

Any number of reviewers can work at once. A reviewer is a name typed on the
first screen; a name not seen before starts a new file. Every decision is
written to that reviewer's own file and folded into one combined file, both
atomically, so a closed tab or a killed server never loses work.

Sample comes from vendor/conv-smi/sample_judge_validation.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parents[1]
SAMPLE_PATH = Path(
    os.environ.get(
        "JUDGE_REVIEW_SAMPLE",
        REPO_ROOT / "vendor" / "conv-smi" / "human_validation" / "sample.json",
    )
).resolve()
RUNTIME_DIR = Path(
    os.environ.get("JUDGE_REVIEW_RUNTIME", APP_ROOT / "runtime")
).resolve()
EXPORT_PATH = Path(
    os.environ.get(
        "JUDGE_REVIEW_EXPORT",
        REPO_ROOT / "data" / "derived" / "judge-validation" / "verdicts.json",
    )
).resolve()
STATE_SCHEMA = "layered-vqa-judge-validation-v1"
VERDICTS = ("correct", "wrong", "unclear")

app = FastAPI(title="Layered-VQA Judge Validation", docs_url=None, redoc_url=None)
lock = threading.RLock()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                     dir=path.parent)
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


def load_sample() -> dict[str, Any]:
    if not SAMPLE_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=("No sample found. Run "
                    "`python sample_judge_validation.py` in vendor/conv-smi "
                    "first."),
        )
    return read_json(SAMPLE_PATH, {"items": []})


def slug(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_").lower()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Reviewer name is empty.")
    return cleaned[:40]


def reviewer_path(name: str) -> Path:
    return RUNTIME_DIR / f"verdicts_{slug(name)}.json"


def load_reviewer(name: str) -> dict[str, Any]:
    return read_json(reviewer_path(name), {
        "schema": STATE_SCHEMA,
        "reviewer": name.strip(),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verdicts": {},
    })


def save_reviewer(name: str, record: dict[str, Any]) -> None:
    """Write the reviewer's file, then rebuild the combined export.

    The export is rebuilt from every reviewer file rather than patched, so a
    file edited by hand or copied in from another machine is picked up without
    a separate merge step.
    """
    record["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record["n_done"] = len(record["verdicts"])
    atomic_json(reviewer_path(name), record)

    reviewers = {}
    for path in sorted(RUNTIME_DIR.glob("verdicts_*.json")):
        data = read_json(path, None)
        if isinstance(data, dict) and data.get("reviewer"):
            reviewers[data["reviewer"]] = data
    atomic_json(EXPORT_PATH, {
        "schema": STATE_SCHEMA,
        "sample": SAMPLE_PATH.name,
        "updated_at": record["updated_at"],
        "reviewers": reviewers,
    })


def public_item(item: dict[str, Any]) -> dict[str, Any]:
    """Everything the page needs, and nothing that reveals the judges."""
    return {k: v for k, v in item.items() if not k.startswith("_")}


class Decision(BaseModel):
    reviewer: str = Field(min_length=1)
    index: int
    verdict: str
    note: str = ""


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "sample": SAMPLE_PATH.exists()}


@app.get("/api/session")
def session(reviewer: str) -> dict[str, Any]:
    with lock:
        sample = load_sample()
        items = sample.get("items", [])
        record = load_reviewer(reviewer)
        done = record["verdicts"]
        cursor = next((i for i, it in enumerate(items)
                       if str(it["index"]) not in done), max(0, len(items) - 1))
        others = []
        for path in sorted(RUNTIME_DIR.glob("verdicts_*.json")):
            data = read_json(path, None)
            if isinstance(data, dict) and data.get("reviewer"):
                others.append({"reviewer": data["reviewer"],
                               "n_done": len(data.get("verdicts", {}))})
        return {
            "reviewer": record["reviewer"],
            "total": len(items),
            "done": len(done),
            "cursor": cursor,
            "verdicts": done,
            "reviewers": others,
        }


@app.get("/api/item/{position}")
def item(position: int, reviewer: str) -> dict[str, Any]:
    with lock:
        items = load_sample().get("items", [])
        if not 0 <= position < len(items):
            raise HTTPException(status_code=404, detail="Position out of range.")
        record = load_reviewer(reviewer)
        current = items[position]
        return {
            "position": position,
            "total": len(items),
            "item": public_item(current),
            "existing": record["verdicts"].get(str(current["index"])),
        }


@app.post("/api/decide")
def decide(decision: Decision) -> dict[str, Any]:
    if decision.verdict not in VERDICTS:
        raise HTTPException(status_code=400, detail="Unknown verdict.")
    with lock:
        items = load_sample().get("items", [])
        match = next((it for it in items if it["index"] == decision.index), None)
        if match is None:
            raise HTTPException(status_code=404, detail="Unknown item.")
        record = load_reviewer(decision.reviewer)
        record["verdicts"][str(decision.index)] = {
            "conv_id": match["conv_id"],
            "condition": match["condition"],
            "verdict": decision.verdict,
            "note": decision.note.strip(),
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        save_reviewer(decision.reviewer, record)
        try:
            shown = EXPORT_PATH.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            shown = str(EXPORT_PATH)
        return {"done": len(record["verdicts"]), "total": len(items),
                "export": shown}


@app.get("/api/export/download")
def download() -> FileResponse:
    if not EXPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="Nothing exported yet.")
    return FileResponse(EXPORT_PATH, filename="judge_validation_verdicts.json",
                        media_type="application/json")


@app.get("/api/image/{relative_path:path}")
def image(relative_path: str) -> FileResponse:
    normalized = relative_path.replace("\\", "/").lstrip("/")
    if not normalized.startswith("data/"):
        raise HTTPException(status_code=403,
                            detail="Only repository data images are servable.")
    candidate = (REPO_ROOT / normalized).resolve()
    data_root = (REPO_ROOT / "data").resolve()
    try:
        candidate.relative_to(data_root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Invalid image path.") from exc
    if candidate.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"} \
            or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(candidate)


app.mount("/", StaticFiles(directory=APP_ROOT / "static", html=True), name="static")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8794)
    args = parser.parse_args()
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
