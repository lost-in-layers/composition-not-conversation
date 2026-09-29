"""Deterministic, persistent, per-reviewer review queues.

The selected scenes are split **disjointly** between the two reviewer slots so
each reviewer owns a fixed set of images and the two together cover the whole
selection exactly once:

* within each subset (LVIS, LAION) the scene ids are shuffled with a fixed seed
  (default 20260719), then split even/odd — ``r1`` takes indices 0, 2, 4… and
  ``r2`` takes 1, 3, 5…; and
* each reviewer's LVIS and LAION halves are interleaved one-by-one (LVIS first)
  to form that reviewer's queue order.

The assignment is a pure function of the index + seed, so the same scene always
lands with the same reviewer. Each reviewer's order is written once to
``runtime/review_queue_<slot>.json`` and is NEVER regenerated on restart.
"""
from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import reviewers
from .atomicio import atomic_write_json, read_json
from .paths import RepoPaths

DEFAULT_SEED = 20260719
QUEUE_SCHEMA_VERSION = 2


def _interleave(a: List[str], b: List[str]) -> List[str]:
    out: List[str] = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            out.append(a[i])
        if i < len(b):
            out.append(b[i])
    return out


def assignments(index: Dict[str, Any], seed: int = DEFAULT_SEED) -> Dict[str, List[str]]:
    """Return ``{slot: [scene_id, ...]}`` — the fixed disjoint split of the
    selected scenes across the two reviewer slots."""
    subsets = index.get("subsets") or {}
    halves: Dict[str, Dict[str, List[str]]] = {slot: {"lvis": [], "laion": []} for slot in reviewers.REVIEWER_SLOTS}
    for subset in ("lvis", "laion"):
        ids = list((subsets.get(subset) or {}).get("scene_ids") or [])
        random.Random(seed).shuffle(ids)
        halves["r1"][subset] = ids[0::2]
        halves["r2"][subset] = ids[1::2]
    return {
        slot: _interleave(halves[slot]["lvis"], halves[slot]["laion"])
        for slot in reviewers.REVIEWER_SLOTS
    }


def build_queue_order(index: Dict[str, Any], reviewer: str, seed: int = DEFAULT_SEED) -> List[str]:
    return assignments(index, seed).get(reviewer, [])


def load_or_create_queue(
    paths: RepoPaths, index: Dict[str, Any], reviewer: str, seed: int = DEFAULT_SEED
) -> Dict[str, Any]:
    queue_path = paths.review_queue_for(reviewer)
    existing = read_json(queue_path)
    if existing and existing.get("schema_version") == QUEUE_SCHEMA_VERSION and existing.get("order"):
        return existing
    order = build_queue_order(index, reviewer, seed)
    queue = {
        "schema_version": QUEUE_SCHEMA_VERSION,
        "seed": seed,
        "reviewer": reviewer,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "order": order,
        "counts": {
            "total": len(order),
            "lvis": sum(1 for sid in order if sid.startswith("lvis_")),
            "laion": sum(1 for sid in order if sid.startswith("laion_")),
        },
    }
    atomic_write_json(queue_path, queue, indent=None)
    return queue


def queue_position(queue: Dict[str, Any], scene_id: str) -> Optional[int]:
    try:
        return queue["order"].index(scene_id)
    except ValueError:
        return None


def next_unfinished(
    queue: Dict[str, Any],
    state: Dict[str, Any],
    start: int,
    direction: int = 1,
    skip_deferred: bool = True,
) -> Optional[str]:
    """Find the nearest scene at/after *start* (or before, direction=-1) that
    has no final decision. Deferred scenes are skipped when requested unless
    everything else is exhausted."""
    order: List[str] = queue.get("order") or []
    decisions = state.get("decisions") or {}
    deferred = set(state.get("deferred") or [])
    n = len(order)
    if n == 0:
        return None

    fallback: Optional[str] = None
    i = start
    while 0 <= i < n:
        sid = order[i]
        if sid not in decisions:
            if sid in deferred and skip_deferred:
                if fallback is None:
                    fallback = sid
            else:
                return sid
        i += direction
    return fallback
