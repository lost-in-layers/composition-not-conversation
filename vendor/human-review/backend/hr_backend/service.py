"""Application service: glue between the HTTP layer and the file-based
storage. Also used directly by the test-suite."""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import curation, exports, indexer, missing, progress, queueing, reviewers, selection
from .atomicio import atomic_write_json, read_json
from .paths import RepoPaths
from .state import StateStore


class ReviewService:
    def __init__(self, paths: Optional[RepoPaths] = None, seed: int = queueing.DEFAULT_SEED) -> None:
        self.paths = paths or RepoPaths()
        self.paths.ensure_runtime_dirs()
        self.seed = seed
        self.lock = threading.RLock()
        self.index: Optional[Dict[str, Any]] = indexer.load_index(self.paths)
        self.reviewer: Optional[str] = None
        self.queue: Optional[Dict[str, Any]] = None
        self.store: Optional[StateStore] = None
        remembered = self._remembered_reviewer()
        if remembered:
            self.set_active_reviewer(remembered)

    # -- reviewer selection ---------------------------------------------
    def _remembered_reviewer(self) -> Optional[str]:
        data = read_json(self.paths.active_reviewer_file) or {}
        slot = data.get("reviewer")
        return slot if reviewers.is_valid_slot(slot or "") else None

    def set_active_reviewer(self, reviewer: str) -> Dict[str, Any]:
        slot = reviewers.normalize_slot(reviewer)
        with self.lock:
            self.reviewer = slot
            self.store = StateStore(self.paths, slot)
            if self.store.state.pop("rebuild_exports_after_migration", False):
                exports.write_exports(self.paths, self.store.state, slot)
                self.store.save()
            self.queue = (
                queueing.load_or_create_queue(self.paths, self.index, slot, self.seed)
                if self.index
                else None
            )
            atomic_write_json(self.paths.active_reviewer_file, {"reviewer": slot})
            return {"reviewer": slot, "reviewer_id": reviewers.REVIEWER_ID[slot]}

    def clear_active_reviewer(self) -> None:
        with self.lock:
            self.reviewer = None
            self.store = None
            self.queue = None
            atomic_write_json(self.paths.active_reviewer_file, {"reviewer": None})

    def _reviewer_summaries(self) -> List[Dict[str, Any]]:
        """Per-slot totals for the reviewer picker (assigned / reviewed)."""
        summaries: List[Dict[str, Any]] = []
        assigned = queueing.assignments(self.index, self.seed) if self.index else {}
        for slot in reviewers.REVIEWER_SLOTS:
            state = read_json(self.paths.review_state_for(slot)) or {}
            decisions = state.get("decisions") or {}
            summaries.append(
                {
                    "reviewer": slot,
                    "reviewer_id": reviewers.REVIEWER_ID[slot],
                    "label": reviewers.REVIEWER_LABEL[slot],
                    "assigned": len(assigned.get(slot, [])),
                    "reviewed": len(decisions),
                    "accepted": sum(1 for d in decisions.values() if d.get("decision") == "accepted"),
                    "rejected": sum(1 for d in decisions.values() if d.get("decision") == "rejected"),
                }
            )
        return summaries

    # ------------------------------------------------------------------
    def ensure_ready(self) -> None:
        if not self.index:
            raise RuntimeError("Scene index not built yet. POST /api/index/rebuild first.")
        if not self.reviewer or not self.store or not self.queue:
            raise RuntimeError("No reviewer selected. POST /api/reviewer first.")

    def rebuild_index(self, progress_cb=None) -> Dict[str, Any]:
        with self.lock:
            self.index = indexer.build_cache(self.paths, progress_cb)
            if self.reviewer:
                self.queue = queueing.load_or_create_queue(self.paths, self.index, self.reviewer, self.seed)
            return {
                "scenes": len(self.index["scenes"]),
                "subsets": {k: v["count"] for k, v in self.index["subsets"].items()},
                "selection": self.index.get("selection"),
            }

    def bootstrap(self) -> Dict[str, Any]:
        with self.lock:
            index_ready = bool(self.index)
            ready = bool(index_ready and self.reviewer and self.queue)
            warnings: List[str] = []
            if self.index:
                warnings = indexer.source_freshness_warnings(self.paths, self.index)
                warnings += indexer.selection_warnings(self.index)
            guide = self.store.state.get("guide") if self.store else {"do_not_autoshow": False}
            payload: Dict[str, Any] = {
                "index_ready": index_ready,
                "ready": ready,
                "reviewer": self.reviewer,
                "reviewer_id": reviewers.REVIEWER_ID[self.reviewer] if self.reviewer else None,
                "guide": guide or {"do_not_autoshow": False},
                "warnings": warnings,
                "seed": (self.queue or {}).get("seed", self.seed),
            }
            if index_ready:
                payload["reviewers"] = self._reviewer_summaries()
            if ready:
                payload["progress"] = progress.compute_progress(self.store.state, self.queue)
                payload["current_scene_id"] = self.current_scene_id()
            if not index_ready:
                payload["sources"] = {
                    subset: {"exists": path.exists(), "path": self.paths.to_repo_relative(path) if path.exists() else str(path.name)}
                    for subset, path in self.paths.sources.items()
                }
                payload["sources"]["selected_1200"] = {
                    "exists": selection.has_selection(self.paths),
                    "path": "data/selected_1200",
                    "count": selection.total_selected(self.paths),
                }
            return payload

    # ------------------------------------------------------------------
    def current_scene_id(self) -> Optional[str]:
        self.ensure_ready()
        state = self.store.state
        cursor = state.get("cursor", 0)
        return queueing.next_unfinished(self.queue, state, cursor, 1)

    def neighbor_scene_id(self, scene_id: str, direction: int) -> Optional[str]:
        self.ensure_ready()
        pos = queueing.queue_position(self.queue, scene_id)
        if pos is None:
            return None
        return queueing.next_unfinished(self.queue, self.store.state, pos + direction, direction)

    def advance_cursor_past(self, scene_id: str) -> None:
        pos = queueing.queue_position(self.queue, scene_id)
        state = self.store.state
        if pos is not None and pos >= state.get("cursor", 0):
            self.store.set_cursor(pos + 1)

    # ------------------------------------------------------------------
    def get_scene(self, scene_id: str, check_files: bool = True) -> Dict[str, Any]:
        self.ensure_ready()
        with self.lock:
            scene = indexer.fetch_scene(self.paths, self.index, scene_id)
            draft = self.store.get_draft(scene_id) or {}
            decision = self.store.decision_for(scene_id)
            review_overlay = draft or (decision or {}).get("review") or {}
            report = missing.classify_scene(scene, review_overlay, self.paths, check_files=check_files)
            suggestions = curation.suggest_categories(scene, review_overlay)
            pos = queueing.queue_position(self.queue, scene_id)
            file_checks = self._file_checks(scene) if check_files else {}
            return {
                "scene": scene,
                "draft": draft or None,
                "decision": decision,
                "status": self.store.scene_status(scene_id),
                "missing_fields": report,
                "category_suggestions": suggestions,
                "queue_position": pos,
                "queue_total": len(self.queue["order"]),
                "file_checks": file_checks,
                "prev_scene_id": self.neighbor_scene_id(scene_id, -1),
                "next_scene_id": self.neighbor_scene_id(scene_id, 1),
            }

    def _file_checks(self, scene: Dict[str, Any]) -> Dict[str, bool]:
        checks: Dict[str, bool] = {}
        for key, rel in (
            ("full", scene.get("full_path")),
            ("background", (scene.get("background") or {}).get("path")),
        ):
            checks[key] = bool(rel) and (self.paths.repo_root / rel).exists()
        for layer in scene.get("layers") or []:
            rel = layer.get("path")
            checks[f"layer_{layer.get('layer_id')}"] = bool(rel) and (self.paths.repo_root / rel).exists()
        strip = (scene.get("render_dir") or "") + "/progressive_strip.png"
        checks["progressive_strip"] = bool(scene.get("render_dir")) and (self.paths.repo_root / strip).exists()
        return checks

    # ------------------------------------------------------------------
    def save_draft(self, scene_id: str, draft: Dict[str, Any]) -> Dict[str, Any]:
        self.ensure_ready()
        with self.lock:
            if scene_id not in self.index["scenes"]:
                raise KeyError(f"unknown scene_id: {scene_id}")
            if self.store.decision_for(scene_id):
                raise ValueError("Scene already has a final decision. Undo it before editing.")
            self.store.save_draft(scene_id, draft)
            scene = indexer.fetch_scene(self.paths, self.index, scene_id)
            report = missing.classify_scene(scene, draft, self.paths, check_files=False)
            return {
                "saved": True,
                "missing_fields": report,
                "category_suggestions": curation.suggest_categories(scene, draft),
            }

    def defer_scene(self, scene_id: str) -> Dict[str, Any]:
        self.ensure_ready()
        with self.lock:
            if self.store.decision_for(scene_id):
                raise ValueError("Scene already finalized; undo first.")
            self.store.defer(scene_id)
            self.advance_cursor_past(scene_id)
            return {"deferred": True, "next_scene_id": self.current_scene_id()}

    def finalize(self, scene_id: str, review: Dict[str, Any], decision: str) -> Dict[str, Any]:
        self.ensure_ready()
        with self.lock:
            if decision not in ("accepted", "rejected"):
                raise ValueError("decision must be 'accepted' or 'rejected'")
            if self.store.decision_for(scene_id):
                raise ValueError("Scene already has a final decision. Undo it first.")
            scene = indexer.fetch_scene(self.paths, self.index, scene_id)
            review = dict(review)
            review.setdefault("reviewer_id", self.store.state.get("reviewer_id"))
            review["finalized_at"] = datetime.now(timezone.utc).isoformat()
            report = missing.classify_scene(scene, review, self.paths, check_files=True)

            record: Dict[str, Any] = {"decision": decision, "review": review, "subset": scene.get("subset")}
            if decision == "accepted":
                problems = curation.validate_acceptance(scene, review)
                blocking = [b for b in report["blocking"] if not self._blocking_resolved(b, review)]
                if blocking:
                    problems.extend(
                        f"Blocking missing source value: {item['field']}"
                        + (f" (layer {item['layer_id']})" if "layer_id" in item else "")
                        + f" — {item['detail']}"
                        for item in blocking
                    )
                if problems:
                    return {"ok": False, "problems": problems}
                curated = curation.build_curated_scene(scene, review, report)
                curated_problems = curation.validate_curated_scene(curated)
                if curated_problems:
                    return {"ok": False, "problems": curated_problems}
                record["curated_scene"] = curated
            else:
                problems = curation.validate_rejection(review)
                if problems:
                    return {"ok": False, "problems": problems}
                record["rejected_record"] = curation.build_rejected_record(scene, review, report)

            self.store.record_decision(scene_id, record)
            self.advance_cursor_past(scene_id)
            export_info = exports.write_exports(self.paths, self.store.state, self.reviewer)
            return {
                "ok": True,
                "decision": decision,
                "export": export_info,
                "next_scene_id": self.current_scene_id(),
                "progress": progress.compute_progress(self.store.state, self.queue),
            }

    @staticmethod
    def _blocking_resolved(item: Dict[str, Any], review: Dict[str, Any]) -> bool:
        """A blocking finding is resolved when the affected layer was excluded
        by the reviewer, or when a reviewer-provided value covers it."""
        layer_id = item.get("layer_id")
        if layer_id is not None:
            reviews = review.get("layer_reviews") or {}
            lr = reviews.get(str(layer_id)) or reviews.get(layer_id) or {}
            if lr.get("decision") == "exclude":
                return True
            if item["field"] == "captions" and (lr.get("caption") or "").strip():
                return True
            return False
        if item["field"] == "captions":
            return bool(
                (review.get("reviewed_image_caption") or "").strip()
                or (review.get("reviewed_image_caption_llava") or "").strip()
            )
        if item["field"] == "background.captions":
            return bool(
                (review.get("reviewed_background_caption") or "").strip()
                or (review.get("reviewed_background_caption_llava") or "").strip()
            )
        return False

    def undo_last(self) -> Dict[str, Any]:
        self.ensure_ready()
        with self.lock:
            entry = self.store.undo_last_decision()
            if entry is None:
                return {"ok": False, "problems": ["Nothing to undo."]}
            export_info = exports.write_exports(self.paths, self.store.state, self.reviewer)
            pos = queueing.queue_position(self.queue, entry["scene_id"])
            if pos is not None and pos < self.store.state.get("cursor", 0):
                self.store.set_cursor(pos)
            return {
                "ok": True,
                "scene_id": entry["scene_id"],
                "export": export_info,
                "progress": progress.compute_progress(self.store.state, self.queue),
            }

    # ------------------------------------------------------------------
    def get_progress(self) -> Dict[str, Any]:
        self.ensure_ready()
        return progress.compute_progress(self.store.state, self.queue)

    def get_dashboard(self) -> Dict[str, Any]:
        self.ensure_ready()
        return {
            "progress": progress.compute_progress(self.store.state, self.queue),
            "dashboard": progress.compute_dashboard(self.store.state, self.queue),
        }

    def export_now(self) -> Dict[str, Any]:
        self.ensure_ready()
        with self.lock:
            return exports.write_exports(self.paths, self.store.state, self.reviewer)

    def backup_now(self) -> Dict[str, Any]:
        with self.lock:
            target = self.store.backup()
            return {"backup_path": target.name, "directory": str(target.parent)}

    def list_deferred(self) -> List[Dict[str, Any]]:
        self.ensure_ready()
        out = []
        for sid in self.store.state.get("deferred") or []:
            meta = self.index["scenes"].get(sid) or {}
            out.append({"scene_id": sid, "subset": meta.get("subset"), "n_layers": meta.get("n_layers")})
        return out
