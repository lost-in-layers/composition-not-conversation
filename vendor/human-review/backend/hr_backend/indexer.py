"""Stream the giant MuLAn source JSON files once with ijson, writing one
compact scene per line to a local JSONL cache plus a byte-offset index so any
scene can later be fetched with a single seek. No database is used.

The cache is rebuildable at any time without touching review state.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

import ijson

from . import selection
from .atomicio import atomic_write_json, read_json
from .paths import RepoPaths

INDEX_SCHEMA_VERSION = 2


def _source_fingerprint(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {"path": path.name, "size_bytes": stat.st_size, "mtime": stat.st_mtime}


def iter_scenes(source_path: Path) -> Iterator[Dict[str, Any]]:
    """Yield scene dicts from a source file without loading the whole file."""
    with open(source_path, "rb") as handle:
        yield from ijson.items(handle, "scenes.item", use_float=True)


def read_source_header(source_path: Path) -> Dict[str, Any]:
    """Read top-level metadata fields that appear before the scenes array."""
    header: Dict[str, Any] = {}
    wanted = {"schema_version", "subset", "path_policy", "mask_policy", "geometry_policy", "source_inventory"}
    with open(source_path, "rb") as handle:
        try:
            for key, value in ijson.kvitems(handle, "", use_float=True):
                if key in wanted:
                    header[key] = value
                if key == "scenes":  # pragma: no cover - kvitems skips arrays lazily
                    break
                if len(header) == len(wanted):
                    break
        except ijson.common.IncompleteJSONError:  # pragma: no cover
            pass
    return header


def build_cache(
    paths: RepoPaths,
    progress: Optional[Callable[[str, int], None]] = None,
) -> Dict[str, Any]:
    """(Re)build the JSONL caches and scene index from the source files.

    Only scenes present under ``data/selected_1200`` are cached, and each
    scene's render paths are repointed to that curated copy (see
    :mod:`hr_backend.selection`). Review state files are never touched.
    Returns the new index payload.
    """
    paths.ensure_runtime_dirs()

    selected = selection.selected_scene_ids(paths)
    if not any(selected.values()):
        raise FileNotFoundError(
            f"No selected scenes found under {paths.selected_dir}. Expected "
            "data/selected_1200/<subset>/<image_id>/full.png for the reviewable "
            "subset."
        )

    index: Dict[str, Any] = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "sources": {},
        "subsets": {},
        "scenes": {},
        "selection": {"root": "data/selected_1200", "expected": {}, "found": {}, "missing_ids": {}},
    }

    for subset, source_path in paths.sources.items():
        if not source_path.exists():
            raise FileNotFoundError(
                f"Source file missing: {source_path}. Expected repo-relative "
                f"data/derived/mulan-scenes/{subset}_scenes.json"
            )
        selected_ids = selected.get(subset, set())
        index["selection"]["expected"][subset] = len(selected_ids)
        cache_path = paths.cache_jsonl[subset]
        ordered_ids = []
        found_ids: set = set()
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{cache_path.name}.", suffix=".tmp", dir=str(cache_path.parent)
        )
        count = 0
        try:
            with os.fdopen(fd, "wb") as out:
                offset = 0
                for scene in iter_scenes(source_path):
                    scene_id = scene["scene_id"]
                    if scene_id not in selected_ids:
                        continue  # not part of the curated selected_1200 subset
                    selection.repoint_scene(scene)
                    line = json.dumps(scene, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                    out.write(line)
                    out.write(b"\n")
                    layers = scene.get("layers") or []
                    index["scenes"][scene_id] = {
                        "file": cache_path.name,
                        "subset": subset,
                        "offset": offset,
                        "length": len(line),
                        "image_id": scene.get("image_id"),
                        "n_layers": len(layers),
                        "width": scene.get("width"),
                        "height": scene.get("height"),
                    }
                    ordered_ids.append(scene_id)
                    found_ids.add(scene_id)
                    offset += len(line) + 1
                    count += 1
                    if progress and count % 200 == 0:
                        progress(subset, count)
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp_name, cache_path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        missing = sorted(selected_ids - found_ids)
        index["subsets"][subset] = {
            "scene_ids": ordered_ids,
            "count": len(ordered_ids),
            "header": read_source_header(source_path),
        }
        index["sources"][subset] = _source_fingerprint(source_path)
        index["selection"]["found"][subset] = len(found_ids)
        index["selection"]["missing_ids"][subset] = missing
        if progress:
            progress(subset, count)

    atomic_write_json(paths.scene_index, index, indent=None)
    return index


def load_index(paths: RepoPaths) -> Optional[Dict[str, Any]]:
    index = read_json(paths.scene_index)
    if not index or index.get("schema_version") != INDEX_SCHEMA_VERSION:
        return None
    return index


def source_freshness_warnings(paths: RepoPaths, index: Dict[str, Any]) -> list:
    """Warn (never fail) if a source file changed since the cache was built."""
    warnings = []
    for subset, recorded in (index.get("sources") or {}).items():
        source_path = paths.sources.get(subset)
        if source_path is None or not source_path.exists():
            warnings.append(f"Source file for subset '{subset}' is missing on disk.")
            continue
        stat = source_path.stat()
        if stat.st_size != recorded.get("size_bytes") or abs(stat.st_mtime - recorded.get("mtime", 0)) > 1e-6:
            warnings.append(
                f"Source file for subset '{subset}' changed since the cache was built "
                f"(size {recorded.get('size_bytes')} -> {stat.st_size}). "
                "Rebuild the cache from the header menu; review decisions are preserved."
            )
    return warnings


def selection_warnings(index: Dict[str, Any]) -> list:
    """Warn (never fail) when a selected scene had no record in the source JSON,
    i.e. it is present under data/selected_1200 but absent from the scene JSONs
    (so it cannot be reviewed until the metadata is added)."""
    warnings = []
    sel = index.get("selection") or {}
    for subset, missing in (sel.get("missing_ids") or {}).items():
        if missing:
            sample = ", ".join(missing[:5]) + ("…" if len(missing) > 5 else "")
            warnings.append(
                f"{len(missing)} selected '{subset}' scene(s) are missing from "
                f"{subset}_scenes.json and cannot be reviewed (e.g. {sample}). "
                "Add their records to data/derived/mulan-scenes/ and rebuild the index."
            )
    return warnings


def fetch_scene(paths: RepoPaths, index: Dict[str, Any], scene_id: str) -> Dict[str, Any]:
    """Fetch one scene by seeking straight into the JSONL cache."""
    entry = index["scenes"].get(scene_id)
    if entry is None:
        raise KeyError(f"unknown scene_id: {scene_id}")
    cache_path = paths.cache_dir / entry["file"]
    with open(cache_path, "rb") as handle:
        handle.seek(entry["offset"])
        raw = handle.read(entry["length"])
    scene = json.loads(raw.decode("utf-8"))
    if scene.get("scene_id") != scene_id:
        raise RuntimeError(
            f"Cache is stale: expected {scene_id} at offset {entry['offset']} "
            f"but found {scene.get('scene_id')!r}. Rebuild the cache."
        )
    return scene
