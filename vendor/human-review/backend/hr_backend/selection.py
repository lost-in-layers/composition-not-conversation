"""Restrict the review to ``data/selected_1200`` and serve its images.

The pilot is reviewed against a curated subset of renders that lives, one
directory per scene, under::

    data/selected_1200/<subset>/<image_id>/full.png, layer_*.png, progressive_strip.png

``image_id`` here equals the scene records' ``image_id``, so the scene id is
``f"{subset}_{image_id}"`` (e.g. ``laion_000000193``). Two things are derived
from this directory:

* the *selection* — only scenes whose renders are present here are indexed,
  queued and reviewable; and
* the *image location* — a scene's stored render paths (which point at the
  original ``data/pre-data/renders`` tree) are rewritten to point at the copies
  under ``data/selected_1200`` so the app serves the curated images.

The original scene-record JSONs (``data/derived/mulan-scenes/*.json``) remain
the source of every non-image field: captions, labels, geometry, relations and
categories. ``selected_1200`` carries no such metadata.
"""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Dict, List, Set

from .paths import RepoPaths

SUBSETS: tuple[str, ...] = ("lvis", "laion")


def selected_sample_ids(paths: RepoPaths) -> Dict[str, List[str]]:
    """Return ``{subset: [image_id, ...]}`` for every scene present under
    ``data/selected_1200`` (a scene counts only when its ``full.png`` exists)."""
    out: Dict[str, List[str]] = {}
    for subset in SUBSETS:
        base = paths.selected_dir / subset
        ids: List[str] = []
        if base.is_dir():
            for child in sorted(base.iterdir()):
                if child.is_dir() and (child / "full.png").is_file():
                    ids.append(child.name)
        out[subset] = ids
    return out


def selected_scene_ids(paths: RepoPaths) -> Dict[str, Set[str]]:
    """Return ``{subset: {scene_id, ...}}`` for the selected scenes."""
    return {
        subset: {f"{subset}_{image_id}" for image_id in ids}
        for subset, ids in selected_sample_ids(paths).items()
    }


def has_selection(paths: RepoPaths) -> bool:
    return any(selected_sample_ids(paths).values())


def total_selected(paths: RepoPaths) -> int:
    return sum(len(v) for v in selected_sample_ids(paths).values())


def _repoint(base: str, path: Any) -> Any:
    """Rewrite an original render path to its ``selected_1200`` copy, keeping
    only the filename (``full.png``, ``layer_3.png``, …)."""
    if not path or not isinstance(path, str):
        return path
    return f"{base}/{PurePosixPath(path.replace(chr(92), '/')).name}"


def repoint_scene(scene: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite every image path in *scene* to its ``data/selected_1200`` copy.

    Mutates and returns *scene*. Non-image fields are untouched.
    """
    subset = scene.get("subset")
    image_id = scene.get("image_id")
    if not subset or not image_id:
        return scene
    base = f"data/selected_1200/{subset}/{image_id}"

    scene["render_dir"] = base
    if scene.get("full_path"):
        scene["full_path"] = _repoint(base, scene["full_path"])
    else:
        scene["full_path"] = f"{base}/full.png"

    background = scene.get("background")
    if isinstance(background, dict) and background.get("path"):
        background["path"] = _repoint(base, background["path"])

    for layer in scene.get("layers") or []:
        if not isinstance(layer, dict):
            continue
        if layer.get("path"):
            layer["path"] = _repoint(base, layer["path"])
        mask = layer.get("mask")
        if isinstance(mask, dict) and mask.get("path"):
            mask["path"] = _repoint(base, mask["path"])

    inventory = scene.get("inventory")
    if isinstance(inventory, dict) and inventory.get("layer_paths_numeric_order"):
        inventory["layer_paths_numeric_order"] = [
            _repoint(base, p) for p in inventory["layer_paths_numeric_order"]
        ]
    return scene
