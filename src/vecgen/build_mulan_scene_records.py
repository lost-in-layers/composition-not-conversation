"""Build one complete, readable MuLAn scene-record JSON per collection.

The builder treats each existing RGBA instance layer's alpha channel as the
canonical mask. It does not create duplicate mask PNGs or embed mask pixels in
JSON. Run it from the repository root after the enriched inventories and
render stacks have been verified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.vecgen.mulan_dataset_tool import resolve_path


SCHEMA_VERSION = "1.0"
LAYER_NAME = re.compile(r"layer_(-?\d+)\.png")


def json_cell(row: dict[str, str], field: str, default: Any) -> Any:
    text = row.get(field, "")
    if not text:
        return default
    return json.loads(text)


def path_reference(path: Path, root: Path) -> str:
    """Return a portable POSIX path relative to root when possible."""
    resolved = path.resolve()
    try:
        reference = Path(os.path.relpath(resolved, root.resolve()))
    except ValueError:
        reference = resolved
    return reference.as_posix()


def rounded(value: float) -> float:
    return round(float(value), 8)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def connected_components_4(mask: np.ndarray) -> int:
    """Count 4-connected components using row runs and a small union-find."""
    if not np.any(mask):
        return 0

    padded = np.pad(mask, ((0, 0), (1, 1)), constant_values=False)
    transitions = np.diff(padded.astype(np.int8), axis=1)
    start_y, start_x = np.nonzero(transitions == 1)
    end_y, end_x = np.nonzero(transitions == -1)

    parent: list[int] = []

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    previous: list[tuple[int, int, int]] = []
    previous_row = -2
    position = 0
    while position < len(start_y):
        row = int(start_y[position])
        if row != previous_row + 1:
            previous = []
        stop = position
        current: list[tuple[int, int, int]] = []
        while stop < len(start_y) and int(start_y[stop]) == row:
            run_id = len(parent)
            parent.append(run_id)
            current.append((int(start_x[stop]), int(end_x[stop]), run_id))
            stop += 1

        if previous:
            previous_index = 0
            for start, end, run_id in current:
                while (
                    previous_index < len(previous)
                    and previous[previous_index][1] <= start
                ):
                    previous_index += 1
                candidate = previous_index
                while candidate < len(previous) and previous[candidate][0] < end:
                    union(run_id, previous[candidate][2])
                    candidate += 1

        previous = current
        previous_row = row
        position = stop

    return len({find(run_id) for run_id in range(len(parent))})


def mask_geometry(
    alpha: np.ndarray,
    support_threshold: int,
    core_threshold: int,
) -> tuple[dict[str, Any], np.ndarray]:
    height, width = alpha.shape
    support = alpha >= support_threshold
    core = alpha >= core_threshold
    y_coordinates, x_coordinates = np.nonzero(support)
    if not len(x_coordinates):
        raise ValueError("alpha mask has no support pixels")

    x1 = int(x_coordinates.min())
    y1 = int(y_coordinates.min())
    x2 = int(x_coordinates.max()) + 1
    y2 = int(y_coordinates.max()) + 1
    support_pixels = int(support.sum())
    core_pixels = int(core.sum())
    weights = alpha.astype(np.float64) / 255.0
    alpha_area = float(weights.sum())

    x_axis = np.arange(width, dtype=np.float64)
    y_axis = np.arange(height, dtype=np.float64)
    weighted_x = float((weights.sum(axis=0) * x_axis).sum() / alpha_area)
    weighted_y = float((weights.sum(axis=1) * y_axis).sum() / alpha_area)
    support_x = float(x_coordinates.mean())
    support_y = float(y_coordinates.mean())

    horizontal_edges = int(np.count_nonzero(support[:, 1:] != support[:, :-1]))
    vertical_edges = int(np.count_nonzero(support[1:, :] != support[:-1, :]))
    border_edges = int(
        support[0, :].sum()
        + support[-1, :].sum()
        + support[:, 0].sum()
        + support[:, -1].sum()
    )
    bbox_area = (x2 - x1) * (y2 - y1)

    geometry = {
        "bbox_xyxy": [x1, y1, x2, y2],
        "bbox_normalized": [
            rounded(x1 / width),
            rounded(y1 / height),
            rounded(x2 / width),
            rounded(y2 / height),
        ],
        "support_centroid_xy": [rounded(support_x), rounded(support_y)],
        "support_centroid_normalized": [
            rounded(support_x / width),
            rounded(support_y / height),
        ],
        "alpha_weighted_centroid_xy": [rounded(weighted_x), rounded(weighted_y)],
        "alpha_weighted_centroid_normalized": [
            rounded(weighted_x / width),
            rounded(weighted_y / height),
        ],
        "support_area_pixels": support_pixels,
        "support_area_fraction": rounded(support_pixels / (height * width)),
        "core_area_pixels": core_pixels,
        "core_area_fraction": rounded(core_pixels / (height * width)),
        "alpha_area_pixels": rounded(alpha_area),
        "alpha_area_fraction": rounded(alpha_area / (height * width)),
        "bbox_occupancy": rounded(support_pixels / bbox_area),
        "perimeter_pixels_4": horizontal_edges + vertical_edges + border_edges,
        "connected_components_4": connected_components_4(support),
        "border_contact": {
            "left": bool(support[:, 0].any()),
            "top": bool(support[0, :].any()),
            "right": bool(support[:, -1].any()),
            "bottom": bool(support[-1, :].any()),
        },
    }
    geometry["touches_image_border"] = any(geometry["border_contact"].values())
    return geometry, support


def pairwise_geometry(
    layers: list[dict[str, Any]],
    supports: list[np.ndarray],
    image_width: int,
    image_height: int,
    near_threshold: float,
    containment_threshold: float,
    occlusion_min_overlap: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    diagonal = math.hypot(image_width, image_height)

    def add_relation(
        subject: int,
        predicate: str,
        object_id: int,
        score: float,
        source: str,
    ) -> None:
        relations.append({
            "subject_layer_id": subject,
            "predicate": predicate,
            "object_layer_id": object_id,
            "score": rounded(score),
            "source": source,
        })

    for left_index in range(len(layers)):
        for right_index in range(left_index + 1, len(layers)):
            left = layers[left_index]
            right = layers[right_index]
            left_mask = supports[left_index]
            right_mask = supports[right_index]
            intersection = int(np.count_nonzero(left_mask & right_mask))
            left_area = int(left["geometry"]["support_area_pixels"])
            right_area = int(right["geometry"]["support_area_pixels"])
            union = left_area + right_area - intersection
            iou = intersection / union if union else 0.0
            overlap_left = intersection / left_area if left_area else 0.0
            overlap_right = intersection / right_area if right_area else 0.0

            left_center = left["geometry"]["alpha_weighted_centroid_xy"]
            right_center = right["geometry"]["alpha_weighted_centroid_xy"]
            center_distance = math.hypot(
                left_center[0] - right_center[0],
                left_center[1] - right_center[1],
            )
            normalized_distance = center_distance / diagonal

            left_bbox = left["geometry"]["bbox_xyxy"]
            right_bbox = right["geometry"]["bbox_xyxy"]
            pair = {
                "layer_ids": [left["layer_id"], right["layer_id"]],
                "intersection_pixels": intersection,
                "union_pixels": union,
                "mask_iou": rounded(iou),
                "overlap_fraction_first": rounded(overlap_left),
                "overlap_fraction_second": rounded(overlap_right),
                "centroid_distance_pixels": rounded(center_distance),
                "centroid_distance_normalized": rounded(normalized_distance),
            }
            pairs.append(pair)

            if intersection:
                add_relation(
                    left["layer_id"], "overlaps", right["layer_id"], iou, "mask_geometry"
                )
                add_relation(
                    right["layer_id"], "overlaps", left["layer_id"], iou, "mask_geometry"
                )
            if overlap_right >= containment_threshold:
                add_relation(
                    left["layer_id"], "contains", right["layer_id"], overlap_right,
                    "mask_geometry",
                )
            if overlap_left >= containment_threshold:
                add_relation(
                    right["layer_id"], "contains", left["layer_id"], overlap_left,
                    "mask_geometry",
                )

            if left_bbox[2] <= right_bbox[0]:
                gap = (right_bbox[0] - left_bbox[2]) / image_width
                add_relation(
                    left["layer_id"], "left_of", right["layer_id"], 1.0 - gap,
                    "nonoverlapping_bboxes",
                )
                add_relation(
                    right["layer_id"], "right_of", left["layer_id"], 1.0 - gap,
                    "nonoverlapping_bboxes",
                )
            elif right_bbox[2] <= left_bbox[0]:
                gap = (left_bbox[0] - right_bbox[2]) / image_width
                add_relation(
                    right["layer_id"], "left_of", left["layer_id"], 1.0 - gap,
                    "nonoverlapping_bboxes",
                )
                add_relation(
                    left["layer_id"], "right_of", right["layer_id"], 1.0 - gap,
                    "nonoverlapping_bboxes",
                )

            if left_bbox[3] <= right_bbox[1]:
                gap = (right_bbox[1] - left_bbox[3]) / image_height
                add_relation(
                    left["layer_id"], "above", right["layer_id"], 1.0 - gap,
                    "nonoverlapping_bboxes",
                )
                add_relation(
                    right["layer_id"], "below", left["layer_id"], 1.0 - gap,
                    "nonoverlapping_bboxes",
                )
            elif right_bbox[3] <= left_bbox[1]:
                gap = (left_bbox[1] - right_bbox[3]) / image_height
                add_relation(
                    right["layer_id"], "above", left["layer_id"], 1.0 - gap,
                    "nonoverlapping_bboxes",
                )
                add_relation(
                    left["layer_id"], "below", right["layer_id"], 1.0 - gap,
                    "nonoverlapping_bboxes",
                )

            if normalized_distance <= near_threshold:
                score = 1.0 - (normalized_distance / near_threshold)
                add_relation(
                    left["layer_id"], "near", right["layer_id"], score,
                    "centroid_distance",
                )
                add_relation(
                    right["layer_id"], "near", left["layer_id"], score,
                    "centroid_distance",
                )

            lower, higher = left, right
            lower_overlap = overlap_left
            if left["z_index"] > right["z_index"]:
                lower, higher = right, left
                lower_overlap = overlap_right
            if lower_overlap >= occlusion_min_overlap:
                add_relation(
                    higher["layer_id"],
                    "possible_occludes",
                    lower["layer_id"],
                    lower_overlap,
                    "mask_overlap_and_stack_order",
                )

    return pairs, relations


def build_scene_record(
    row: dict[str, str],
    collection: str,
    inventory_dir: Path,
    path_root: Path,
    support_threshold: int,
    core_threshold: int,
    near_threshold: float,
    containment_threshold: float,
    occlusion_min_overlap: float,
    verify_composite: bool,
) -> dict[str, Any]:
    image_id = row.get("image_id", "")
    scene_id = f"{collection}_{image_id}"
    errors: list[str] = []
    warnings: list[str] = []

    try:
        if row.get("render_status") not in {"existing", "rendered"}:
            errors.append(
                f"render_status is {row.get('render_status')!r}, expected existing or rendered"
            )
        instance_keys = json_cell(row, "instance_keys_json", [])
        instance_keys = sorted((str(key) for key in instance_keys), key=int)
        instance_captions = json_cell(row, "instance_blip2_json", {})
        captioning = json_cell(row, "annotation_captioning_json", {})
        annotation_manifest = json_cell(row, "annotation_metadata_json", {})
        expected_layers = int(row.get("object_layers") or 0)
        if len(instance_keys) != expected_layers:
            errors.append(
                f"instance key count {len(instance_keys)} != object_layers {expected_layers}"
            )

        render_dir = resolve_path(row["render_dir"], inventory_dir)
        full_path = resolve_path(row["full_path"], inventory_dir)
        if not render_dir.is_dir():
            raise FileNotFoundError(f"render directory not found: {render_dir}")
        if not full_path.is_file():
            raise FileNotFoundError(f"full render not found: {full_path}")

        actual_layer_paths: dict[int, Path] = {}
        for candidate in render_dir.glob("layer_*.png"):
            match = LAYER_NAME.fullmatch(candidate.name)
            if match:
                actual_layer_paths[int(match.group(1))] = candidate
        expected_layer_ids = [0] + [int(key) for key in instance_keys]
        if set(actual_layer_paths) != set(expected_layer_ids):
            errors.append(
                "render layer IDs "
                f"{sorted(actual_layer_paths)} != expected {expected_layer_ids}"
            )

        with Image.open(full_path) as full_image:
            full_rgb = np.asarray(full_image.convert("RGB")).copy()
        image_height, image_width = full_rgb.shape[:2]

        background_path = actual_layer_paths.get(0, render_dir / "layer_0.png")
        with Image.open(background_path) as background_image:
            if background_image.size != (image_width, image_height):
                errors.append(
                    f"background size {background_image.size} != full size "
                    f"{(image_width, image_height)}"
                )
            canvas = background_image.convert("RGBA").copy()

        native_instances = annotation_manifest.get("instances", {})
        layer_records: list[dict[str, Any]] = []
        supports: list[np.ndarray] = []
        for z_index, key in enumerate(instance_keys, start=1):
            layer_id = int(key)
            layer_path = actual_layer_paths.get(
                layer_id,
                render_dir / f"layer_{key}.png",
            )
            if not layer_path.is_file():
                errors.append(f"missing instance layer: {layer_path}")
                continue

            with Image.open(layer_path) as layer_image:
                if "A" not in layer_image.getbands():
                    errors.append(f"instance layer has no alpha channel: {layer_path}")
                    continue
                if layer_image.size != (image_width, image_height):
                    errors.append(
                        f"layer {layer_id} size {layer_image.size} != full size "
                        f"{(image_width, image_height)}"
                    )
                    continue
                rgba = layer_image.convert("RGBA").copy()
                alpha = np.asarray(rgba.getchannel("A")).copy()

            geometry, support = mask_geometry(
                alpha,
                support_threshold,
                core_threshold,
            )
            alpha_sha256 = hashlib.sha256(alpha.tobytes()).hexdigest()
            native_alpha = native_instances.get(key, {}).get("instance_alpha", {})
            native_sha256 = native_alpha.get("sha256", "")
            if native_alpha.get("shape") not in (None, [image_height, image_width]):
                errors.append(
                    f"layer {layer_id} native alpha shape {native_alpha.get('shape')} "
                    f"!= {[image_height, image_width]}"
                )
            if native_alpha.get("mode") not in (None, "L"):
                errors.append(
                    f"layer {layer_id} native alpha mode {native_alpha.get('mode')!r} != 'L'"
                )
            if native_alpha.get("dtype") not in (None, "uint8"):
                errors.append(
                    f"layer {layer_id} native alpha dtype "
                    f"{native_alpha.get('dtype')!r} != 'uint8'"
                )
            native_match = not native_sha256 or native_sha256 == alpha_sha256
            if not native_sha256:
                warnings.append(f"layer {layer_id} has no native alpha checksum")
            elif not native_match:
                errors.append(f"layer {layer_id} alpha checksum does not match .p.zl manifest")

            layer_records.append({
                "layer_id": layer_id,
                "native_instance_key": key,
                "role": "instance",
                "z_index": z_index,
                "path": path_reference(layer_path, path_root),
                "captions": {
                    "blip2": instance_captions.get(key, ""),
                },
                "label": None,
                "label_source": None,
                "caption_quality": "unverified",
                "mask": {
                    "encoding": "rgba_alpha",
                    "path": path_reference(layer_path, path_root),
                    "channel": "alpha",
                    "width": image_width,
                    "height": image_height,
                    "dtype": "uint8",
                    "support_threshold": support_threshold,
                    "core_threshold": core_threshold,
                    "alpha_sha256": alpha_sha256,
                    "native_alpha_sha256": native_sha256 or None,
                    "matches_native_alpha": native_match if native_sha256 else None,
                },
                "geometry": geometry,
            })
            supports.append(support)
            canvas = Image.alpha_composite(canvas, rgba)

        pairs, relations = pairwise_geometry(
            layer_records,
            supports,
            image_width,
            image_height,
            near_threshold,
            containment_threshold,
            occlusion_min_overlap,
        )

        composite_validation: dict[str, Any] = {"performed": verify_composite}
        if verify_composite and len(layer_records) == len(instance_keys):
            composite_rgb = np.asarray(canvas.convert("RGB"))
            different_channels = composite_rgb != full_rgb
            mismatched = int(np.count_nonzero(np.any(different_channels, axis=2)))
            maximum_delta = 0
            if mismatched:
                maximum_delta = int(
                    np.abs(
                        composite_rgb.astype(np.int16) - full_rgb.astype(np.int16)
                    ).max()
                )
            composite_validation.update({
                "matches_full_png": mismatched == 0,
                "mismatched_pixels": mismatched,
                "maximum_channel_delta": maximum_delta,
            })
            if mismatched:
                errors.append(
                    f"ordered layer composite differs from full.png at {mismatched} pixels"
                )

        pzl_path = resolve_path(row["pzl_path"], inventory_dir)
        if not pzl_path.is_file():
            errors.append(f"native annotation not found: {pzl_path}")
            actual_pzl_size = None
            actual_pzl_sha256 = None
        else:
            actual_pzl_size = pzl_path.stat().st_size
            actual_pzl_sha256 = file_sha256(pzl_path)
            expected_pzl_size = int(row.get("annotation_size_bytes") or 0)
            expected_pzl_sha256 = row.get("annotation_sha256", "")
            if expected_pzl_size and actual_pzl_size != expected_pzl_size:
                errors.append(
                    f"native annotation size {actual_pzl_size} != inventory "
                    f"size {expected_pzl_size}"
                )
            if expected_pzl_sha256 and actual_pzl_sha256 != expected_pzl_sha256:
                errors.append("native annotation SHA-256 does not match inventory")
        categories = [
            category for category in row.get("categories", "").split("|") if category
        ]
        return {
            "scene_id": scene_id,
            "subset": collection,
            "render_subset": row.get("subset", ""),
            "image_id": image_id,
            "width": image_width,
            "height": image_height,
            "full_path": path_reference(full_path, path_root),
            "render_dir": path_reference(render_dir, path_root),
            "image_ref": row.get("image_ref", ""),
            "categories": categories,
            "category_scope": "image_level_candidates_not_per_layer_labels",
            "captions": {
                "blip2": row.get("image_blip2", ""),
                "llava": row.get("image_llava", ""),
            },
            "captioning_models": captioning,
            "annotation": {
                "pzl_path": path_reference(pzl_path, path_root),
                "sha256": row.get("annotation_sha256", ""),
                "size_bytes": int(row.get("annotation_size_bytes") or 0),
                "actual_sha256": actual_pzl_sha256,
                "actual_size_bytes": actual_pzl_size,
                "matches_inventory_sha256": (
                    actual_pzl_sha256 == row.get("annotation_sha256", "")
                    if actual_pzl_sha256 and row.get("annotation_sha256", "")
                    else None
                ),
                "native_manifest": annotation_manifest,
            },
            "inventory": {
                "render_status": row.get("render_status", ""),
                "object_layers": expected_layers,
                "total_layers_including_background": int(
                    row.get("total_layers_including_background") or 0
                ),
                "instance_keys": instance_keys,
                "layer_paths_numeric_order": [
                    path_reference(background_path, path_root),
                    *[layer["path"] for layer in layer_records],
                ],
                "error": row.get("error", ""),
            },
            "background": {
                "layer_id": 0,
                "role": "background",
                "z_index": 0,
                "path": path_reference(background_path, path_root),
                "captions": {
                    "blip2": row.get("background_blip2", ""),
                    "llava": row.get("background_llava", ""),
                },
            },
            "layers": layer_records,
            "pairwise_geometry": pairs,
            "relations": relations,
            "validation": {
                "status": "valid" if not errors else "invalid",
                "errors": errors,
                "warnings": warnings,
                "composite": composite_validation,
            },
        }
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        return {
            "scene_id": scene_id,
            "subset": collection,
            "image_id": image_id,
            "validation": {
                "status": "invalid",
                "errors": errors,
                "warnings": warnings,
                "composite": {"performed": False},
            },
        }


def inventory_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json_property(handle: Any, name: str, value: Any, comma: bool) -> None:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).splitlines()
    handle.write(f"  {json.dumps(name)}: {encoded[0]}")
    for line in encoded[1:]:
        handle.write(f"\n  {line}")
    handle.write(",\n" if comma else "\n")


def ordered_scene_records(
    rows: list[dict[str, str]],
    workers: int,
    builder: Any,
) -> Iterable[dict[str, Any]]:
    batch_size = max(workers * 4, 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(rows), batch_size):
            futures = [pool.submit(builder, row) for row in rows[start : start + batch_size]]
            for future in futures:
                yield future.result()


def exclusion_reason(scene: dict[str, Any]) -> str:
    errors = scene.get("validation", {}).get("errors", [])
    if any(error.startswith("render_status") for error in errors):
        return "missing_or_unusable_render"
    if any("alpha mask has no support pixels" in error for error in errors):
        return "empty_alpha_mask"
    return "validation_failed"


def build_collection(
    collection: str,
    inventory_path: Path,
    output_path: Path,
    path_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    rows = inventory_rows(inventory_path)
    if not rows:
        raise RuntimeError(f"Inventory contains no rows: {inventory_path}")
    duplicate_ids = [
        image_id
        for image_id, count in Counter(row.get("image_id", "") for row in rows).items()
        if count > 1
    ]
    if duplicate_ids:
        raise RuntimeError(
            f"Inventory has {len(duplicate_ids)} duplicate image IDs; "
            f"first duplicate: {sorted(duplicate_ids)[0]}"
        )
    rows.sort(key=lambda row: row.get("image_id", ""))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    source_status_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    total_layers = 0
    included_scenes = 0
    excluded_scenes: list[dict[str, Any]] = []

    def builder(row: dict[str, str]) -> dict[str, Any]:
        return build_scene_record(
            row,
            collection,
            inventory_path.resolve().parent,
            path_root,
            args.support_threshold,
            args.core_threshold,
            args.near_threshold,
            args.containment_threshold,
            args.occlusion_min_overlap,
            not args.skip_composite_validation,
        )

    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("{\n")
            write_json_property(handle, "schema_version", SCHEMA_VERSION, True)
            write_json_property(handle, "subset", collection, True)
            write_json_property(
                handle,
                "path_policy",
                {
                    "all_paths_are_relative_to": "path_root",
                    "path_root_from_json_directory": path_reference(
                        path_root,
                        output_path.resolve().parent,
                    ),
                },
                True,
            )
            write_json_property(
                handle,
                "source_inventory",
                path_reference(inventory_path, path_root),
                True,
            )
            write_json_property(
                handle,
                "mask_policy",
                {
                    "encoding": "rgba_alpha",
                    "duplicate_mask_pngs_created": False,
                    "inline_rle_created": False,
                    "support_threshold": args.support_threshold,
                    "core_threshold": args.core_threshold,
                },
                True,
            )
            write_json_property(
                handle,
                "geometry_policy",
                {
                    "bbox_format": "half_open_xyxy",
                    "support_definition": f"alpha >= {args.support_threshold}",
                    "core_definition": f"alpha >= {args.core_threshold}",
                    "centroid_coordinates": "zero_based_pixel_indices",
                    "component_connectivity": 4,
                    "z_index_meaning": (
                        "ordinal alpha-composite order; larger is on top, not metric depth"
                    ),
                    "near_centroid_distance_normalized_threshold": args.near_threshold,
                    "containment_overlap_fraction_threshold": args.containment_threshold,
                    "possible_occlusion_lower_mask_overlap_threshold": (
                        args.occlusion_min_overlap
                    ),
                },
                True,
            )
            handle.write('  "scenes": [\n')

            records = ordered_scene_records(rows, args.workers, builder)
            for index, scene in enumerate(
                tqdm(records, total=len(rows), desc=f"Build {collection}", unit="scene")
            ):
                status = scene["validation"]["status"]
                source_status_counts[status] += 1
                if status == "invalid" and args.exclude_invalid:
                    excluded_scenes.append({
                        "scene_id": scene.get("scene_id", ""),
                        "image_id": scene.get("image_id", ""),
                        "reason": exclusion_reason(scene),
                        "errors": scene["validation"].get("errors", []),
                    })
                    continue

                if included_scenes:
                    handle.write(",\n")
                encoded = json.dumps(
                    scene,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                handle.write("\n".join(f"    {line}" for line in encoded.splitlines()))
                included_scenes += 1
                total_layers += len(scene.get("layers", []))
                relation_counts.update(
                    relation["predicate"] for relation in scene.get("relations", [])
                )

            handle.write("\n  ],\n")
            write_json_property(
                handle,
                "excluded_scenes",
                excluded_scenes,
                True,
            )
            invalid_in_output = (
                0 if args.exclude_invalid else source_status_counts["invalid"]
            )
            exclusion_counts = Counter(
                item["reason"] for item in excluded_scenes
            )
            summary = {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_inventory_scenes": len(rows),
                "total_scenes": included_scenes,
                "valid_scenes": source_status_counts["valid"],
                "invalid_scenes": invalid_in_output,
                "excluded_invalid_scenes": len(excluded_scenes),
                "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
                "total_object_layers": total_layers,
                "relation_counts": dict(sorted(relation_counts.items())),
            }
            write_json_property(handle, "summary", summary, False)
            handle.write("}\n")
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    print(f"Written: {output_path}")
    print(
        f"Scenes: source={len(rows)}, included={included_scenes}, "
        f"valid={source_status_counts['valid']}, "
        f"invalid={source_status_counts['invalid']}, "
        f"excluded={len(excluded_scenes)}"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build one indented JSON scene-record file each for MuLAn LVIS and LAION."
        )
    )
    parser.add_argument(
        "--inventory-root",
        default="data/pre-data/inventories",
        help="Directory containing lvis.csv and laion.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/derived/mulan-scenes",
        help="Directory for lvis_scenes.json and laion_scenes.json.",
    )
    parser.add_argument(
        "--path-root",
        default=".",
        help="Root used to make paths inside JSON portable and relative.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--support-threshold", type=int, default=1)
    parser.add_argument("--core-threshold", type=int, default=128)
    parser.add_argument("--near-threshold", type=float, default=0.1)
    parser.add_argument("--containment-threshold", type=float, default=0.98)
    parser.add_argument("--occlusion-min-overlap", type=float, default=0.01)
    parser.add_argument(
        "--skip-composite-validation",
        action="store_true",
        help="Skip reconstructing full.png from the numerically ordered layers.",
    )
    parser.add_argument(
        "--exclude-invalid",
        action="store_true",
        help=(
            "Omit invalid scenes from the main scenes array while retaining "
            "their IDs and errors in excluded_scenes."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace existing output JSON files.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if not 1 <= args.support_threshold <= 255:
        raise ValueError("--support-threshold must be between 1 and 255")
    if not 1 <= args.core_threshold <= 255:
        raise ValueError("--core-threshold must be between 1 and 255")
    if args.core_threshold < args.support_threshold:
        raise ValueError("--core-threshold must be >= --support-threshold")
    for name in ("near_threshold", "containment_threshold", "occlusion_min_overlap"):
        value = getattr(args, name)
        if not 0.0 < value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in (0, 1]")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    inventory_root = Path(args.inventory_root)
    output_dir = Path(args.output_dir)
    path_root = Path(args.path_root)

    summaries: dict[str, dict[str, Any]] = {}
    for collection in ("lvis", "laion"):
        inventory_path = inventory_root / f"{collection}.csv"
        if not inventory_path.is_file():
            raise FileNotFoundError(f"Inventory not found: {inventory_path}")
        summaries[collection] = build_collection(
            collection,
            inventory_path,
            output_dir / f"{collection}_scenes.json",
            path_root,
            args,
        )

    invalid = sum(summary["invalid_scenes"] for summary in summaries.values())
    if invalid:
        raise RuntimeError(
            f"Built both JSON files, but {invalid} scenes are invalid. "
            "Search each file for validation.status = invalid."
        )
    print("All MuLAn scene records are valid.")


if __name__ == "__main__":
    main()
