"""The two fixed human reviewers.

Exactly two people review the pilot set; each is a fixed *slot* with its own
persistent state, its own review queue, and its own export files. The 1200
selected scenes are split deterministically and disjointly between the two
slots (see :func:`hr_backend.queueing.assignments`), so a given scene always
belongs to the same reviewer.
"""
from __future__ import annotations

from typing import List

# Internal slot ids (used in filenames and the API).
REVIEWER_SLOTS: tuple[str, ...] = ("r1", "r2")

# Stable identifier recorded inside decisions and export files.
REVIEWER_ID = {"r1": "reviewer_1", "r2": "reviewer_2"}

# Human-facing label shown in the UI.
REVIEWER_LABEL = {"r1": "Reviewer 1", "r2": "Reviewer 2"}


def is_valid_slot(slot: str) -> bool:
    return slot in REVIEWER_SLOTS


def slot_number(slot: str) -> int:
    """1 for ``r1``, 2 for ``r2``."""
    return REVIEWER_SLOTS.index(slot) + 1


def normalize_slot(value: str) -> str:
    """Accept ``r1``/``r2``, ``reviewer_1``/``reviewer_2``, ``1``/``2``.

    Raises ``ValueError`` for anything else so a bad request is a clean 400.
    """
    v = (value or "").strip().lower()
    aliases = {
        "r1": "r1", "reviewer_1": "r1", "reviewer 1": "r1", "1": "r1",
        "r2": "r2", "reviewer_2": "r2", "reviewer 2": "r2", "2": "r2",
    }
    slot = aliases.get(v)
    if slot is None:
        raise ValueError(f"unknown reviewer: {value!r} (expected one of {list(REVIEWER_SLOTS)})")
    return slot


def all_slots() -> List[str]:
    return list(REVIEWER_SLOTS)
