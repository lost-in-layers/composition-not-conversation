"""
MuLAn inventory, statistics, and rendering tool.

The inventory CSV is the single source of truth:
  1. `inventory` scans .p.zl annotations and records layer counts and image sources.
  2. `stats` reads the inventory and writes the layer distribution and summary.
  3. `render` reads the inventory and saves only:
       layer_0.png        background RGB
       layer_1.png ...    instance RGBA layers
       full.png           all layers composited

No progressive steps or strips are saved.

Examples are provided in the accompanying response.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import pickle
import statistics
import sys
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
COCO_URLS = (
    "https://images.cocodataset.org/train2017/{image_id}.jpg",
    "https://images.cocodataset.org/val2017/{image_id}.jpg",
)

INVENTORY_FIELDS = [
    "subset",
    "image_id",
    "pzl_path",
    "annotation_sha256",
    "annotation_size_bytes",
    "annotation_captioning_json",
    "annotation_metadata_json",
    "image_blip2",
    "image_llava",
    "background_blip2",
    "background_llava",
    "instance_keys_json",
    "instance_blip2_json",
    "image_ref",
    "categories",
    "object_layers",
    "total_layers_including_background",
    "render_status",
    "render_dir",
    "full_path",
    "layer_paths",
    "error",
]


def path_reference(path: Path, base_dir: Path) -> str:
    """Store a relative path when possible, otherwise an absolute path."""
    path = path.resolve()
    base_dir = base_dir.resolve()
    try:
        return os.path.relpath(path, base_dir)
    except ValueError:
        return str(path)


def resolve_path(value: str, base_dir: Path) -> Path:
    """Resolve inventory paths relative to the inventory CSV directory."""
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_pzl(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return pickle.loads(zlib.decompress(file.read()))


def json_text(value: Any) -> str:
    """Serialize compact, deterministic UTF-8 JSON for a CSV cell."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def native_value_manifest(value: Any) -> Any:
    """Create a lossless-annotation manifest without duplicating array bytes.

    The original compressed `.p.zl` remains the canonical lossless payload.
    Scalars and text are copied into the manifest. Large/binary values are
    represented by type, shape/size, and SHA-256 so the inventory can prove
    exactly which native payload it references without ballooning the CSV.
    """
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "__native_type__": "numpy.ndarray",
            "dtype": str(contiguous.dtype),
            "shape": list(contiguous.shape),
            "nbytes": int(contiguous.nbytes),
            "sha256": hashlib.sha256(
                memoryview(contiguous).cast("B")
            ).hexdigest(),
        }

    if isinstance(value, Image.Image):
        array = np.ascontiguousarray(np.asarray(value))
        return {
            "__native_type__": "PIL.Image.Image",
            "mode": value.mode,
            "size": list(value.size),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "nbytes": int(array.nbytes),
            "sha256": hashlib.sha256(
                memoryview(array).cast("B")
            ).hexdigest(),
        }

    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        return {
            "__native_type__": type(value).__name__,
            "nbytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    if isinstance(value, np.generic):
        return native_value_manifest(value.item())

    if isinstance(value, dict):
        return {
            str(key): native_value_manifest(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [native_value_manifest(item) for item in value]

    if isinstance(value, set):
        return sorted(
            (native_value_manifest(item) for item in value),
            key=lambda item: json_text(item),
        )

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, float) and not math.isfinite(value):
        return {
            "__native_type__": "float",
            "value": str(value),
        }

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    return {
        "__native_type__": (
            f"{type(value).__module__}.{type(value).__qualname__}"
        ),
        "repr": repr(value),
    }


def caption_text(container: Any, key: str) -> str:
    """Return a caption as text while preserving non-string values as JSON."""
    if not isinstance(container, dict) or key not in container:
        return ""
    value = container[key]
    if isinstance(value, str):
        return value
    return json_text(native_value_manifest(value))


def inspect_pzl(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    compressed = path.read_bytes()
    annotation = pickle.loads(zlib.decompress(compressed))
    instances = annotation.get("instances")
    if instances is None:
        raise KeyError("Missing 'instances' field")

    image = annotation.get("image", {})
    background = annotation.get("background", {})
    captioning = annotation.get("captioning", {})
    instance_keys = sorted(instances, key=lambda key: int(key))
    instance_blip2 = {
        str(key): caption_text(instances[key], "blip2")
        for key in instance_keys
    }
    object_layers = len(instances)
    return {
        "image_id": path.name.removesuffix(".p.zl"),
        "pzl_path": str(path.resolve()),
        "annotation_sha256": hashlib.sha256(compressed).hexdigest(),
        "annotation_size_bytes": len(compressed),
        "annotation_captioning_json": json_text(
            native_value_manifest(captioning)
        ),
        "annotation_metadata_json": json_text(
            native_value_manifest(annotation)
        ),
        "image_blip2": caption_text(image, "blip2"),
        "image_llava": caption_text(image, "llava"),
        "background_blip2": caption_text(background, "blip2"),
        "background_llava": caption_text(background, "llava"),
        "instance_keys_json": json_text([str(key) for key in instance_keys]),
        "instance_blip2_json": json_text(instance_blip2),
        "object_layers": object_layers,
        "total_layers_including_background": object_layers + 1,
    }


def load_category_index(
    cache_path: Path | None,
    coco_annotations_dir: Path | None,
) -> dict[str, list[str]]:
    if cache_path and cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    if not coco_annotations_dir:
        raise FileNotFoundError(
            "Category cache was not found and --coco-annotations-dir was not provided."
        )

    annotation_files = sorted(
        coco_annotations_dir.glob("instances_*2017.json")
    )
    if not annotation_files:
        raise FileNotFoundError(
            f"No instances_*2017.json files found in {coco_annotations_dir}"
        )

    index: dict[str, set[str]] = {}
    for annotation_path in annotation_files:
        print(f"Reading {annotation_path.name}...")
        with annotation_path.open("r", encoding="utf-8") as file:
            coco = json.load(file)

        category_names = {
            category["id"]: category["name"]
            for category in coco["categories"]
        }
        for annotation in tqdm(
            coco["annotations"],
            desc=annotation_path.name,
            unit="ann",
        ):
            image_id = f"{annotation['image_id']:012d}"
            index.setdefault(image_id, set()).add(
                category_names[annotation["category_id"]]
            )

    serializable = {
        image_id: sorted(categories)
        for image_id, categories in index.items()
    }

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as file:
            json.dump(serializable, file)

    return serializable


def load_url_map(path: Path | None) -> dict[str, str]:
    if not path:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"URL map not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"URL map has no header: {path}")

        fields = reader.fieldnames
        lowered = {field.lower(): field for field in fields}

        id_field = next(
            (
                lowered[name]
                for name in (
                    "image_id",
                    "file_id",
                    "id",
                    "filename",
                    "file",
                    "key",
                )
                if name in lowered
            ),
            None,
        )
        url_field = next(
            (
                field
                for field in fields
                if "url" in field.lower()
            ),
            None,
        )

        if not url_field and len(fields) >= 2:
            url_field = fields[-1]
        if not id_field:
            id_field = next(
                (field for field in fields if field != url_field),
                None,
            )
        if not id_field or not url_field:
            raise ValueError(
                "Could not identify ID and URL columns in "
                f"{path}. Columns: {fields}"
            )

        mapping: dict[str, str] = {}
        for row in reader:
            raw_id = (row.get(id_field) or "").strip()
            url = (row.get(url_field) or "").strip()
            if raw_id and url:
                mapping[Path(raw_id).stem] = url

    return mapping


def build_image_index(images_dir: Path | None) -> dict[str, str]:
    if not images_dir:
        return {}
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_paths = [
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    index: dict[str, str] = {}
    for path in tqdm(image_paths, desc="Indexing source images", unit="img"):
        index.setdefault(path.stem, str(path.resolve()))
    return index


def find_lvis_image(
    image_id: str,
    images_dir: Path | None,
    image_index: dict[str, str],
) -> str:
    if images_dir:
        candidates = (
            images_dir / "train2017" / f"{image_id}.jpg",
            images_dir / "val2017" / f"{image_id}.jpg",
            images_dir / f"{image_id}.jpg",
        )
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())

    return image_index.get(image_id, f"coco://{image_id}")


def select_categories(
    image_id: str,
    category_index: dict[str, list[str]],
    wanted: set[str],
    match_mode: str,
) -> tuple[bool, str]:
    categories = category_index.get(image_id, [])
    if not wanted:
        return True, "|".join(categories)

    predicate = all if match_mode == "all" else any
    selected = predicate(category in categories for category in wanted)
    return selected, "|".join(categories)


def write_inventory(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=INVENTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_inventory(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def verify_inventory_row(
    row: dict[str, str],
    inventory_dir_text: str,
    require_renders: bool,
) -> tuple[str, str]:
    """Verify that a row still points to its exact native annotation/render."""
    inventory_dir = Path(inventory_dir_text)
    image_id = row.get("image_id", "")
    try:
        pzl_path = resolve_path(row["pzl_path"], inventory_dir)
        compressed = pzl_path.read_bytes()

        expected_size = int(row.get("annotation_size_bytes") or 0)
        if expected_size and len(compressed) != expected_size:
            raise ValueError(
                f"annotation size {len(compressed)} != {expected_size}"
            )

        expected_sha256 = row.get("annotation_sha256", "")
        actual_sha256 = hashlib.sha256(compressed).hexdigest()
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise ValueError(
                f"annotation SHA-256 {actual_sha256} != {expected_sha256}"
            )

        annotation = pickle.loads(zlib.decompress(compressed))
        actual_layers = len(annotation.get("instances", {}))
        expected_layers = int(row.get("object_layers") or 0)
        if actual_layers != expected_layers:
            raise ValueError(
                f"object layer count {actual_layers} != {expected_layers}"
            )

        if require_renders:
            full_path_text = row.get("full_path", "")
            if not full_path_text:
                raise ValueError("full_path is empty")
            full_path = resolve_path(full_path_text, inventory_dir)
            if not full_path.is_file():
                raise FileNotFoundError(f"missing full render: {full_path}")

            layer_paths = json.loads(row.get("layer_paths") or "[]")
            expected_total = actual_layers + 1
            if len(layer_paths) != expected_total:
                raise ValueError(
                    f"render layer count {len(layer_paths)} != {expected_total}"
                )
            for layer_path_text in layer_paths:
                layer_path = resolve_path(layer_path_text, inventory_dir)
                if not layer_path.is_file():
                    raise FileNotFoundError(f"missing layer render: {layer_path}")
    except Exception as exc:
        return image_id, f"{type(exc).__name__}: {exc}"

    return image_id, ""


def command_verify(args: argparse.Namespace) -> None:
    inventory_path = Path(args.inventory)
    inventory_dir = inventory_path.resolve().parent
    rows = read_inventory(inventory_path)
    selected = [
        row
        for row in rows
        if int(row["object_layers"]) >= args.min_layers
        and (
            not args.max_layers
            or int(row["object_layers"]) <= args.max_layers
        )
        and (
            not args.render_status
            or row.get("render_status") == args.render_status
        )
    ]
    if args.limit:
        selected = selected[: args.limit]

    if not selected:
        filters = [
            f"min_layers={args.min_layers}",
            f"max_layers={args.max_layers or 'none'}",
            f"render_status={args.render_status or 'any'}",
        ]
        raise RuntimeError(
            "No inventory rows matched the verification filters "
            f"({', '.join(filters)}). Check render_status counts and paths "
            "before treating verification as successful."
        )

    failures: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                verify_inventory_row,
                row,
                str(inventory_dir),
                args.require_renders,
            )
            for row in selected
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Verify inventory",
            unit="row",
        ):
            image_id, error = future.result()
            if error:
                failures.append((image_id, error))

    failure_path = inventory_path.with_name(
        f"{inventory_path.stem}_verification_failures.csv"
    )
    if failures:
        with failure_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["image_id", "error"])
            writer.writerows(sorted(failures))
        raise RuntimeError(
            f"Verification failed for {len(failures)} of {len(selected)} rows. "
            f"See {failure_path}."
        )

    if failure_path.exists():
        failure_path.unlink()
    print(f"Verified: {len(selected)} rows")
    print(f"Inventory: {inventory_path}")
    print(f"Required renders: {args.require_renders}")


def command_inventory(args: argparse.Namespace) -> None:
    pzl_dir = Path(args.pzl_dir)
    inventory_path = Path(args.inventory)
    images_dir = Path(args.images_dir) if args.images_dir else None
    cache_path = Path(args.category_cache) if args.category_cache else None
    coco_annotations_dir = (
        Path(args.coco_annotations_dir)
        if args.coco_annotations_dir
        else None
    )
    url_map_path = Path(args.url_map) if args.url_map else None

    if not pzl_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {pzl_dir}")

    pzl_paths = sorted(pzl_dir.rglob("*.p.zl"))
    if args.limit:
        pzl_paths = pzl_paths[: args.limit]
    if not pzl_paths:
        raise FileNotFoundError(f"No .p.zl files found in {pzl_dir}")

    wanted = {
        category.strip().lower()
        for category in (args.categories or "").split(",")
        if category.strip()
    }
    if wanted and args.subset != "lvis":
        raise ValueError("--categories is supported only for subset 'lvis'.")

    category_index: dict[str, list[str]] = {}
    if args.subset == "lvis" and (wanted or cache_path):
        category_index = load_category_index(
            cache_path,
            coco_annotations_dir,
        )

    image_index = build_image_index(images_dir)
    url_map = load_url_map(url_map_path)

    inspected: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(inspect_pzl, str(path)): path
            for path in pzl_paths
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Inventory {args.subset}",
            unit="file",
        ):
            source = futures[future]
            try:
                inspected.append(future.result())
            except Exception as exc:
                failures.append((str(source), f"{type(exc).__name__}: {exc}"))

    rows: list[dict[str, Any]] = []
    for item in sorted(inspected, key=lambda row: row["image_id"]):
        image_id = item["image_id"]

        selected, categories = select_categories(
            image_id,
            category_index,
            wanted,
            args.match,
        )
        if not selected:
            continue

        if item["object_layers"] < args.min_layers:
            continue
        if args.max_layers and item["object_layers"] > args.max_layers:
            continue

        if args.subset == "lvis":
            image_ref = find_lvis_image(
                image_id,
                images_dir,
                image_index,
            )
        else:
            image_ref = image_index.get(image_id) or url_map.get(image_id, "")

        inventory_dir = inventory_path.resolve().parent
        pzl_ref = path_reference(Path(item["pzl_path"]), inventory_dir)
        if image_ref and not image_ref.startswith(
            ("http://", "https://", "coco://")
        ):
            image_ref = path_reference(Path(image_ref), inventory_dir)

        rows.append({
            "subset": args.subset,
            "image_id": image_id,
            "pzl_path": pzl_ref,
            "annotation_sha256": item["annotation_sha256"],
            "annotation_size_bytes": item["annotation_size_bytes"],
            "annotation_captioning_json": (
                item["annotation_captioning_json"]
            ),
            "annotation_metadata_json": item["annotation_metadata_json"],
            "image_blip2": item["image_blip2"],
            "image_llava": item["image_llava"],
            "background_blip2": item["background_blip2"],
            "background_llava": item["background_llava"],
            "instance_keys_json": item["instance_keys_json"],
            "instance_blip2_json": item["instance_blip2_json"],
            "image_ref": image_ref,
            "categories": categories,
            "object_layers": item["object_layers"],
            "total_layers_including_background": (
                item["total_layers_including_background"]
            ),
            "render_status": "",
            "render_dir": "",
            "full_path": "",
            "layer_paths": "",
            "error": "",
        })

    write_inventory(inventory_path, rows)

    print(f"\nInventory rows: {len(rows)}")
    print(f"Inspection failures: {len(failures)}")
    print(f"Written: {inventory_path}")

    if args.subset != "lvis":
        missing_sources = sum(not row["image_ref"] for row in rows)
        if missing_sources:
            print(
                f"LAION source image missing for {missing_sources} row(s). "
                "Stats still work; rendering those rows requires "
                "--images-dir or --url-map when rebuilding the inventory."
            )

    if failures:
        failure_path = inventory_path.with_name(
            f"{inventory_path.stem}_failures.csv"
        )
        with failure_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["pzl_path", "error"])
            writer.writerows(failures)
        print(f"Failures written: {failure_path}")


def command_stats(args: argparse.Namespace) -> None:
    inventory_path = Path(args.inventory)
    output_dir = Path(args.output_dir)
    rows = read_inventory(inventory_path)

    selected_rows = []
    for row in tqdm(rows, desc="Computing stats", unit="row"):
        count = int(row["object_layers"])
        if count < args.min_layers:
            continue
        if args.max_layers and count > args.max_layers:
            continue
        selected_rows.append(row)

    counts = [int(row["object_layers"]) for row in selected_rows]
    distribution = Counter(counts)
    output_dir.mkdir(parents=True, exist_ok=True)

    distribution_path = output_dir / "layer_distribution.csv"
    with distribution_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "object_layers",
            "total_layers_including_background",
            "main_instances",
        ])
        for layer_count in sorted(distribution):
            writer.writerow([
                layer_count,
                layer_count + 1,
                distribution[layer_count],
            ])

    summary = {
        "inventory": path_reference(inventory_path, Path.cwd()),
        "total_main_instances": len(counts),
        "total_object_layers": sum(counts),
        "total_layers_including_background": sum(
            count + 1 for count in counts
        ),
        "minimum_object_layers": min(counts) if counts else None,
        "maximum_object_layers": max(counts) if counts else None,
        "instances_with_maximum_object_layers": (
            distribution[max(counts)] if counts else 0
        ),
        "mean_object_layers": (
            round(statistics.fmean(counts), 4) if counts else None
        ),
        "median_object_layers": (
            statistics.median(counts) if counts else None
        ),
        "layer_distribution": {
            str(layer_count): distribution[layer_count]
            for layer_count in sorted(distribution)
        },
    }

    summary_path = output_dir / "layer_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nStats complete")
    print(f"Main instances: {summary['total_main_instances']}")
    print(f"Total object layers: {summary['total_object_layers']}")
    print(
        "Object layers: "
        f"min={summary['minimum_object_layers']}, "
        f"max={summary['maximum_object_layers']}, "
        f"mean={summary['mean_object_layers']}, "
        f"median={summary['median_object_layers']}"
    )
    print(
        "Instances with maximum layers: "
        f"{summary['instances_with_maximum_object_layers']}"
    )
    print(f"Written: {distribution_path}")
    print(f"Written: {summary_path}")


def load_base_image(
    image_ref: str,
    image_id: str,
    inventory_dir: Path,
    timeout: int,
) -> np.ndarray:
    if image_ref.startswith(("http://", "https://")):
        response = requests.get(image_ref, timeout=timeout)
        response.raise_for_status()
        return np.asarray(
            Image.open(io.BytesIO(response.content)).convert("RGB")
        )

    if image_ref.startswith("coco://"):
        errors = []
        for template in COCO_URLS:
            url = template.format(image_id=image_id)
            try:
                response = requests.get(url, timeout=timeout)
                if response.ok:
                    return np.asarray(
                        Image.open(io.BytesIO(response.content)).convert("RGB")
                    )
                errors.append(f"{response.status_code} {url}")
            except requests.RequestException as exc:
                errors.append(f"{url}: {exc}")
        raise RuntimeError(
            "COCO image could not be downloaded: " + "; ".join(errors)
        )

    if not image_ref:
        raise FileNotFoundError(
            "No source image is recorded. Rebuild the inventory with "
            "--images-dir or --url-map."
        )

    image_path = resolve_path(image_ref, inventory_dir)
    if not image_path.exists():
        raise FileNotFoundError(f"Source image not found: {image_path}")
    return np.asarray(Image.open(image_path).convert("RGB"))


def reconstruct_background(base: np.ndarray, background: dict[str, Any]) -> Image.Image:
    reconstructed = (
        base * background["original_image_mask"]
        + background["inpainted_delta"]
    )
    return Image.fromarray(
        reconstructed.astype(np.uint8)
    ).convert("RGB")


def reconstruct_instance(base: np.ndarray, instance: dict[str, Any]) -> Image.Image:
    reconstructed = (
        base
        * instance["original_image_mask"]
        * instance["instance_mask"]
        + instance["inpainted_delta"]
    )
    rgba = Image.fromarray(
        reconstructed.astype(np.uint8)
    ).convert("RGB")

    alpha = instance["instance_alpha"]
    if isinstance(alpha, np.ndarray):
        alpha = Image.fromarray(alpha)
    rgba.putalpha(alpha)
    return rgba


def render_one(
    row: dict[str, str],
    inventory_dir_text: str,
    output_dir_text: str,
    timeout: int,
    overwrite: bool,
    existing_only: bool,
) -> dict[str, str]:
    inventory_dir = Path(inventory_dir_text)
    output_root = Path(output_dir_text)
    image_id = row["image_id"]
    subset = row["subset"]

    # Load only the small annotation first.
    pzl_path = resolve_path(row["pzl_path"], inventory_dir)
    annotation = load_pzl(pzl_path)

    instance_dir = output_root / subset / image_id
    full_path = instance_dir / "full.png"
    expected_layer_count = len(annotation["instances"]) + 1

    # Check existing output BEFORE downloading/loading the source image.
    if not overwrite and full_path.exists():
        existing_layers = sorted(instance_dir.glob("layer_*.png"))

        if len(existing_layers) == expected_layer_count:
            return {
                "render_status": "existing",
                "render_dir": path_reference(
                    instance_dir,
                    inventory_dir,
                ),
                "full_path": path_reference(
                    full_path,
                    inventory_dir,
                ),
                "layer_paths": json.dumps([
                    path_reference(path, inventory_dir)
                    for path in existing_layers
                ]),
                "error": "",
            }

    if existing_only:
        return {
            "render_status": "missing",
            "render_dir": "",
            "full_path": "",
            "layer_paths": "",
            "error": "Existing render not found or incomplete.",
        }

    # Only incomplete or missing renders reach this point.
    base = load_base_image(
        row["image_ref"],
        image_id,
        inventory_dir,
        timeout,
    )

    background_mask = annotation["background"]["original_image_mask"]
    if base.shape[:2] != background_mask.shape[:2]:
        raise RuntimeError(
            f"Image/mask shape mismatch: {base.shape[:2]} vs "
            f"{background_mask.shape[:2]}"
        )

    instance_dir.mkdir(parents=True, exist_ok=True)

    layer_paths: list[str] = []

    background = reconstruct_background(
        base,
        annotation["background"],
    )
    background_path = instance_dir / "layer_0.png"
    background.save(background_path)
    layer_paths.append(path_reference(background_path, inventory_dir))

    canvas = background.convert("RGBA")
    for layer_key in sorted(annotation["instances"], key=int):
        instance = reconstruct_instance(
            base,
            annotation["instances"][layer_key],
        )
        layer_path = instance_dir / f"layer_{layer_key}.png"
        instance.save(layer_path)
        layer_paths.append(path_reference(layer_path, inventory_dir))
        canvas = Image.alpha_composite(canvas, instance)

    canvas.convert("RGB").save(full_path)

    return {
        "render_status": "rendered",
        "render_dir": path_reference(instance_dir, inventory_dir),
        "full_path": path_reference(full_path, inventory_dir),
        "layer_paths": json.dumps(layer_paths),
        "error": "",
    }

def command_render(args: argparse.Namespace) -> None:
    inventory_path = Path(args.inventory)
    inventory_dir = inventory_path.resolve().parent
    output_dir = Path(args.output_dir)
    rows = read_inventory(inventory_path)

    selected_indices = [
        index
        for index, row in enumerate(rows)
        if int(row["object_layers"]) >= args.min_layers
        and (
            not args.max_layers
            or int(row["object_layers"]) <= args.max_layers
        )
    ]
    if args.limit:
        selected_indices = selected_indices[: args.limit]

    if not selected_indices:
        raise RuntimeError(
            "No inventory rows matched the render layer filters."
        )

    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                render_one,
                rows[index],
                str(inventory_dir),
                str(output_dir),
                args.timeout,
                args.overwrite,
                args.existing_only,
            ): index
            for index in selected_indices
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Rendering",
            unit="image",
        ):
            index = futures[future]
            try:
                rows[index].update(future.result())
            except Exception as exc:
                failures += 1
                rows[index].update({
                    "render_status": "failed",
                    "render_dir": "",
                    "full_path": "",
                    "layer_paths": "",
                    "error": f"{type(exc).__name__}: {exc}",
                })

    status_counts = Counter(
        rows[index].get("render_status") or "empty"
        for index in selected_indices
    )
    write_inventory(inventory_path, rows)
    print(f"\nRendered/checked: {len(selected_indices)}")
    print(
        "Statuses: "
        + ", ".join(
            f"{status}={count}"
            for status, count in sorted(status_counts.items())
        )
    )
    print(f"Failures: {failures}")
    print(f"Updated inventory: {inventory_path}")

    if args.existing_only and not status_counts.get("existing"):
        raise RuntimeError(
            "No existing render stacks were found. The inventory was updated "
            "with missing statuses; check whether the archive was extracted "
            "with an extra directory level under the render output root."
        )


def add_layer_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--min-layers",
        type=int,
        default=0,
        help="Minimum number of object layers.",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=0,
        help="Maximum object layers; 0 means no maximum.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a MuLAn inventory, calculate layer statistics, "
            "render individual layers plus the final composite, and verify "
            "lossless annotation references."
        )
    )
    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    inventory = commands.add_parser(
        "inventory",
        help="Scan .p.zl files and create the inventory CSV.",
    )
    inventory.add_argument(
        "--subset",
        required=True,
        help="Subset label, such as lvis or laion.",
    )
    inventory.add_argument(
        "--pzl-dir",
        required=True,
        help="Directory containing the subset's .p.zl files.",
    )
    inventory.add_argument(
        "--inventory",
        required=True,
        help="Output inventory CSV path.",
    )
    inventory.add_argument(
        "--images-dir",
        help=(
            "Optional source-image directory. For LVIS this may contain "
            "train2017/ and val2017/. For LAION it is searched recursively."
        ),
    )
    inventory.add_argument(
        "--url-map",
        help="Optional LAION CSV containing image IDs and source URLs.",
    )
    inventory.add_argument(
        "--category-cache",
        help="Optional coco_image_categories.json path.",
    )
    inventory.add_argument(
        "--coco-annotations-dir",
        help=(
            "Optional directory containing instances_train2017.json and "
            "instances_val2017.json; used only if the category cache is missing."
        ),
    )
    inventory.add_argument(
        "--categories",
        help="LVIS-only comma-separated category filter.",
    )
    inventory.add_argument(
        "--match",
        choices=("any", "all"),
        default="all",
        help="Category matching mode.",
    )
    inventory.add_argument(
        "--limit",
        type=int,
        help="Maximum number of .p.zl files to inspect.",
    )
    inventory.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Worker processes.",
    )
    add_layer_filters(inventory)
    inventory.set_defaults(handler=command_inventory)

    stats = commands.add_parser(
        "stats",
        help="Calculate statistics from an inventory CSV.",
    )
    stats.add_argument(
        "--inventory",
        required=True,
        help="Input inventory CSV.",
    )
    stats.add_argument(
        "--output-dir",
        required=True,
        help="Directory for summary JSON and distribution CSV.",
    )
    add_layer_filters(stats)
    stats.set_defaults(handler=command_stats)

    render = commands.add_parser(
        "render",
        help="Save only layer PNGs and full.png.",
    )
    render.add_argument(
        "--inventory",
        required=True,
        help="Inventory CSV to read and update.",
    )
    render.add_argument(
        "--output-dir",
        required=True,
        help="Root directory for rendered output.",
    )
    render.add_argument(
        "--limit",
        type=int,
        help="Maximum number of inventory rows to render.",
    )
    render.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Worker processes.",
    )
    render.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds.",
    )
    render.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite complete existing output.",
    )
    render.add_argument(
        "--existing-only",
        action="store_true",
        help=(
            "Index complete existing renders without downloading source "
            "images or generating missing renders."
        ),
    )
    add_layer_filters(render)
    render.set_defaults(handler=command_render)

    verify = commands.add_parser(
        "verify",
        help="Verify annotation checksums and optional render completeness.",
    )
    verify.add_argument(
        "--inventory",
        required=True,
        help="Inventory CSV to verify.",
    )
    verify.add_argument(
        "--require-renders",
        action="store_true",
        help="Also require full.png and every listed layer PNG.",
    )
    verify.add_argument(
        "--render-status",
        help="Verify only rows with this render_status value.",
    )
    verify.add_argument(
        "--limit",
        type=int,
        help="Maximum number of inventory rows to verify.",
    )
    verify.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Worker processes.",
    )
    add_layer_filters(verify)
    verify.set_defaults(handler=command_verify)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.handler(args)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
