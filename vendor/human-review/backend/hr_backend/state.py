"""File-based review state (no database), one file per reviewer slot.

``runtime/review_state_<slot>.json`` holds that reviewer's queue cursor, drafts,
deferrals, final decisions, undo history and UI preferences. Every write goes
through a temp file + atomic replace.
"""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import reviewers
from .atomicio import atomic_write_json, read_json
from .paths import RepoPaths

STATE_SCHEMA_VERSION = 3
UNDO_DEPTH = 25


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_state(reviewer: str) -> Dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "reviewer_slot": reviewer,
        "reviewer_id": reviewers.REVIEWER_ID[reviewer],
        "guide": {"do_not_autoshow": False},
        "cursor": 0,
        "drafts": {},
        "deferred": [],
        "decisions": {},
        "decision_order": [],
        "undo_history": [],
        "created_at": _now(),
        "updated_at": _now(),
    }


class StateStore:
    def __init__(self, paths: RepoPaths, reviewer: str) -> None:
        self.paths = paths
        self.reviewer = reviewer
        self._path = paths.review_state_for(reviewer)
        self._state: Dict[str, Any] = read_json(self._path) or default_state(reviewer)
        if self._state.get("schema_version") == 2:
            self._migrate_v2_to_v3()
        elif self._state.get("schema_version") != STATE_SCHEMA_VERSION:
            raise RuntimeError(
                f"{self._path.name} has schema_version {self._state.get('schema_version')}; "
                f"this build expects {STATE_SCHEMA_VERSION}."
            )
        # Keep identity fields consistent with the slot that owns this file.
        self._state["reviewer_slot"] = reviewer
        self._state["reviewer_id"] = reviewers.REVIEWER_ID[reviewer]

    def _migrate_v2_to_v3(self) -> None:
        """Reopen accepted scenes that predate required question answers.

        All review values are copied back into drafts. Only accepted decisions
        with no complete answers are reopened; rejections remain final.
        """
        if self._path.exists():
            self.paths.backup_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            shutil.copyfile(self._path, self.paths.backup_dir / f"{self._path.stem}.{stamp}.pre-v3.json")

        decisions = self._state.setdefault("decisions", {})
        drafts = self._state.setdefault("drafts", {})
        reopened: List[str] = []
        for scene_id, record in list(decisions.items()):
            if not isinstance(record, dict) or record.get("decision") != "accepted":
                continue
            review = dict(record.get("review") or {})
            suitable = review.get("suitable_categories") or []
            questions = dict(review.get("human_questions") or {})
            legacy = review.get("human_question")
            primary = review.get("primary_category")
            if not questions and primary in suitable and isinstance(legacy, str) and legacy.strip():
                questions[primary] = legacy.strip()
            answers = dict(review.get("human_answers") or {})
            filled = {
                category: text
                for category, text in questions.items()
                if category in suitable and isinstance(text, str) and text.strip()
            }
            complete = bool(filled) and all(
                isinstance(answers.get(category), str) and answers[category].strip()
                for category in filled
            )
            if complete:
                continue

            review["human_questions"] = questions
            review["human_answers"] = answers
            review["reopened_for_required_answers"] = True
            review.pop("finalized_at", None)
            drafts[scene_id] = review
            decisions.pop(scene_id, None)
            reopened.append(scene_id)

        if reopened:
            self._state["decision_order"] = [
                scene_id for scene_id in (self._state.get("decision_order") or []) if scene_id not in reopened
            ]
            self._state["cursor"] = 0
            self._state["reopened_for_required_answers"] = reopened
            self._state["rebuild_exports_after_migration"] = True
        self._state["schema_version"] = STATE_SCHEMA_VERSION
        self._state["updated_at"] = _now()
        atomic_write_json(self._path, self._state)

    # -- persistence ------------------------------------------------------
    @property
    def state(self) -> Dict[str, Any]:
        return self._state

    def save(self) -> None:
        self._state["updated_at"] = _now()
        atomic_write_json(self._path, self._state)

    def backup(self) -> Path:
        """Create a timestamped copy of the current state file."""
        self.save()
        self.paths.backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = self.paths.backup_dir / f"review_state_{self.reviewer}.{stamp}.json"
        shutil.copyfile(self._path, target)
        return target

    # -- preferences -----------------------------------------------------
    def set_guide_preference(self, do_not_autoshow: bool) -> None:
        self._state.setdefault("guide", {})["do_not_autoshow"] = bool(do_not_autoshow)
        self.save()

    # -- drafts -----------------------------------------------------------
    def save_draft(self, scene_id: str, draft: Dict[str, Any]) -> None:
        draft = dict(draft)
        draft["scene_id"] = scene_id
        draft["updated_at"] = _now()
        self._state["drafts"][scene_id] = draft
        self.save()

    def get_draft(self, scene_id: str) -> Optional[Dict[str, Any]]:
        return self._state["drafts"].get(scene_id)

    def drop_draft(self, scene_id: str) -> None:
        if scene_id in self._state["drafts"]:
            del self._state["drafts"][scene_id]
            self.save()

    # -- deferral ---------------------------------------------------------
    def defer(self, scene_id: str) -> None:
        if scene_id not in self._state["deferred"]:
            self._state["deferred"].append(scene_id)
        self.save()

    def undefer(self, scene_id: str) -> None:
        if scene_id in self._state["deferred"]:
            self._state["deferred"].remove(scene_id)
            self.save()

    # -- cursor -----------------------------------------------------------
    def set_cursor(self, position: int) -> None:
        self._state["cursor"] = max(0, int(position))
        self.save()

    # -- final decisions --------------------------------------------------
    def record_decision(self, scene_id: str, record: Dict[str, Any]) -> None:
        record = dict(record)
        record["scene_id"] = scene_id
        record["decided_at"] = _now()
        self._state["decisions"][scene_id] = record
        order: List[str] = self._state["decision_order"]
        if scene_id in order:
            order.remove(scene_id)
        order.append(scene_id)
        if scene_id in self._state["deferred"]:
            self._state["deferred"].remove(scene_id)
        # The draft that produced the decision is preserved inside the record;
        # remove it from the active drafts map.
        self._state["drafts"].pop(scene_id, None)
        self.save()

    def undo_last_decision(self) -> Optional[Dict[str, Any]]:
        """Move the most recent final decision back into drafts."""
        order: List[str] = self._state["decision_order"]
        if not order:
            return None
        scene_id = order.pop()
        record = self._state["decisions"].pop(scene_id, None)
        if record is None:
            return None
        undo_entry = {"scene_id": scene_id, "record": record, "undone_at": _now()}
        history: List[Dict[str, Any]] = self._state.setdefault("undo_history", [])
        history.append(undo_entry)
        del history[:-UNDO_DEPTH]
        draft = record.get("review")
        if draft:
            draft = dict(draft)
            draft["restored_from_undo"] = True
            self._state["drafts"][scene_id] = draft
        self.save()
        return undo_entry

    # -- convenience ------------------------------------------------------
    def decision_for(self, scene_id: str) -> Optional[Dict[str, Any]]:
        return self._state["decisions"].get(scene_id)

    def scene_status(self, scene_id: str) -> str:
        if scene_id in self._state["decisions"]:
            return self._state["decisions"][scene_id].get("decision", "finalized")
        if scene_id in self._state["deferred"]:
            return "deferred"
        if scene_id in self._state["drafts"]:
            return "draft"
        return "unseen"
