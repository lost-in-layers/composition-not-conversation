"""Repository-relative path resolution and safe path handling.

The repo root is discovered by walking upward from this file until a directory
containing the expected repository markers is found. Nothing is ever hardcoded
to an absolute machine-specific location.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_REPO_MARKERS = (".git", "data")

# Subtrees (relative to repo root) that the image endpoint is allowed to serve.
SERVABLE_SUBTREES = ("data",)
SERVABLE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def find_repo_root(start: Optional[Path] = None) -> Path:
    """Walk upward from *start* (default: this file) to find the repo root."""
    env_root = os.environ.get("HR_REPO_ROOT")
    if env_root:
        root = Path(env_root).resolve()
        if root.is_dir():
            return root
        raise RuntimeError(f"HR_REPO_ROOT points to a non-directory: {env_root}")

    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in _REPO_MARKERS):
            # Require it to actually look like the Layered-VQA repo when the
            # marker was only ``data`` (a bare data folder is not enough).
            if (candidate / ".git").exists() or (candidate / "data" / "derived").exists():
                return candidate
    raise RuntimeError(
        "Could not locate the Layered-VQA repository root. "
        "Set the HR_REPO_ROOT environment variable to the repo checkout."
    )


class RepoPaths:
    """All well-known locations, derived from the repo root."""

    def __init__(self, repo_root: Optional[Path] = None) -> None:
        self.repo_root = (repo_root or find_repo_root()).resolve()
        self.app_root = self.repo_root / "vendor" / "human-review"
        self.runtime_dir = self.app_root / "runtime"
        self.cache_dir = self.runtime_dir / "cache"
        self.backup_dir = self.runtime_dir / "backups"
        self.export_dir = self.repo_root / "data" / "derived" / "mulan-human-review"

        # Curated subset that defines which scenes are reviewable and where
        # their render images live.
        self.selected_dir = self.repo_root / "data" / "selected_1200"

        self.sources = {
            "lvis": self.repo_root / "data" / "derived" / "mulan-scenes" / "lvis_scenes.json",
            "laion": self.repo_root / "data" / "derived" / "mulan-scenes" / "laion_scenes.json",
        }
        self.cache_jsonl = {
            "lvis": self.cache_dir / "lvis_scenes.jsonl",
            "laion": self.cache_dir / "laion_scenes.jsonl",
        }
        self.scene_index = self.cache_dir / "scene_index.json"
        # Remembers which reviewer slot was last active on this machine.
        self.active_reviewer_file = self.runtime_dir / "active_reviewer.json"
        self.frontend_dist = self.app_root / "frontend" / "dist"

    # -- per-reviewer locations ------------------------------------------
    def review_queue_for(self, slot: str) -> Path:
        return self.runtime_dir / f"review_queue_{slot}.json"

    def review_state_for(self, slot: str) -> Path:
        return self.runtime_dir / f"review_state_{slot}.json"

    def export_dir_for(self, slot: str) -> Path:
        from .reviewers import slot_number

        return self.export_dir / f"reviewer_{slot_number(slot)}"

    def accepted_export_for(self, slot: str) -> Path:
        return self.export_dir_for(slot) / "accepted_scenes.json"

    def rejected_export_for(self, slot: str) -> Path:
        return self.export_dir_for(slot) / "rejected_scenes.json"

    def ensure_runtime_dirs(self) -> None:
        for d in (self.runtime_dir, self.cache_dir, self.backup_dir, self.export_dir):
            d.mkdir(parents=True, exist_ok=True)

    def resolve_servable(self, relative_path: str) -> Path:
        """Resolve a repo-relative image path, rejecting traversal/absolute paths.

        Raises ``ValueError`` when the path escapes the allowed subtrees or has
        a non-image suffix. Existence is NOT checked here.
        """
        rel = relative_path.replace("\\", "/").strip()
        if not rel or rel.startswith("/") or ":" in rel.split("/")[0]:
            raise ValueError("absolute paths are not servable")
        candidate = (self.repo_root / rel).resolve()
        allowed = False
        for subtree in SERVABLE_SUBTREES:
            base = (self.repo_root / subtree).resolve()
            try:
                candidate.relative_to(base)
                allowed = True
                break
            except ValueError:
                continue
        if not allowed:
            raise ValueError("path escapes the servable data directory")
        if candidate.suffix.lower() not in SERVABLE_SUFFIXES:
            raise ValueError("unsupported file type")
        return candidate

    def to_repo_relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.repo_root).as_posix()
