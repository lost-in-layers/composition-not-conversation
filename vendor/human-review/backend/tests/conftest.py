"""Fixtures: a miniature fake repository with schema-faithful source files.

The real source files are ~270 MB each; tests always use these tiny fixtures.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from hr_backend.paths import RepoPaths  # noqa: E402
from hr_backend.service import ReviewService  # noqa: E402

# A valid 1x1 transparent PNG.
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

RENDER_SUBDIR = {"lvis": "lvis/lvis", "laion": "laion/laion_aesthetic_v2_6.5"}


def make_layer(render_dir: str, layer_id: int, caption: Optional[str] = "a fixture object") -> Dict[str, Any]:
    captions = {} if caption is None else {"blip2": caption}
    return {
        "layer_id": layer_id,
        "native_instance_key": str(layer_id),
        "role": "instance",
        "z_index": layer_id,
        "path": f"{render_dir}/layer_{layer_id}.png",
        "captions": captions,
        "label": None,
        "label_source": None,
        "caption_quality": "unverified",
        "mask": {
            "encoding": "rgba_alpha",
            "path": f"{render_dir}/layer_{layer_id}.png",
            "channel": "alpha",
            "width": 64,
            "height": 48,
            "native_alpha_sha256": None,
            "matches_native_alpha": None,
        },
        "geometry": {
            "bbox_xyxy": [4 * layer_id, 2 * layer_id, 4 * layer_id + 10, 2 * layer_id + 8],
            "support_centroid_xy": [4.0 * layer_id + 5, 2.0 * layer_id + 4],
            "support_area_pixels": 80,
            "connected_components_4": 1,
            "touches_image_border": False,
        },
    }


def make_scene(
    subset: str,
    image_id: str,
    n_layers: int = 2,
    with_relations: bool = True,
    with_occlusion: bool = False,
    layer_without_caption: Optional[int] = None,
) -> Dict[str, Any]:
    render_dir = f"data/pre-data/renders/{RENDER_SUBDIR[subset]}/{image_id}"
    layers: List[Dict[str, Any]] = []
    for i in range(1, n_layers + 1):
        caption = None if layer_without_caption == i else f"fixture object {i}"
        layers.append(make_layer(render_dir, i, caption))

    relations = []
    pairwise = []
    if with_relations and n_layers >= 2:
        for i in range(1, n_layers):
            relations.append(
                {"subject_layer_id": i, "predicate": "left_of", "object_layer_id": i + 1, "score": 0.9, "source": "nonoverlapping_bboxes"}
            )
            relations.append(
                {"subject_layer_id": i + 1, "predicate": "right_of", "object_layer_id": i, "score": 0.9, "source": "nonoverlapping_bboxes"}
            )
            pairwise.append(
                {
                    "layer_ids": [i, i + 1],
                    "intersection_pixels": 12 if with_occlusion else 0,
                    "union_pixels": 100,
                    "mask_iou": 0.12 if with_occlusion else 0.0,
                }
            )
        if with_occlusion:
            relations.append(
                {"subject_layer_id": 2, "predicate": "possible_occludes", "object_layer_id": 1, "score": 0.4, "source": "mask_overlap_and_stack_order"}
            )

    return {
        "scene_id": f"{subset}_{image_id}",
        "subset": subset,
        "render_subset": RENDER_SUBDIR[subset].split("/")[-1],
        "image_id": image_id,
        "width": 64,
        "height": 48,
        "full_path": f"{render_dir}/full.png",
        "render_dir": render_dir,
        "image_ref": f"fixture://{image_id}",
        "categories": ["giraffe"] if subset == "lvis" else [],
        "category_scope": "image_level_candidates_not_per_layer_labels",
        "captions": {"blip2": "fixture full caption", "llava": "fixture full caption, but longer"},
        "captioning_models": {"blip2": {"hf_id": "fixture/blip2"}},
        "annotation": {
            "pzl_path": f"data/pre-data/mulan_v1/extracted/{subset}/{image_id}.p.zl",
            "sha256": "0" * 64,
            "size_bytes": 123,
            "actual_sha256": None,
            "actual_size_bytes": None,
            "matches_inventory_sha256": None,
        },
        "inventory": {
            "render_status": "existing",
            "object_layers": n_layers,
            "total_layers_including_background": n_layers + 1,
            "instance_keys": [str(i) for i in range(1, n_layers + 1)],
            "layer_paths_numeric_order": [f"{render_dir}/layer_{i}.png" for i in range(0, n_layers + 1)],
            "error": "",
        },
        "background": {
            "layer_id": 0,
            "role": "background",
            "z_index": 0,
            "path": f"{render_dir}/layer_0.png",
            "captions": {"blip2": "fixture background", "llava": "fixture background, but longer"},
        },
        "layers": layers,
        "pairwise_geometry": pairwise,
        "relations": relations,
        "validation": {"status": "valid", "errors": [], "warnings": []},
    }


def write_source_file(path: Path, subset: str, scenes: List[Dict[str, Any]]) -> None:
    payload = {
        "schema_version": "1.0",
        "subset": subset,
        "path_policy": {"all_paths_are_relative_to": "path_root", "path_root_from_json_directory": "../../.."},
        "source_inventory": f"data/pre-data/inventories/{subset}.csv",
        "mask_policy": {"encoding": "rgba_alpha"},
        "geometry_policy": {"bbox_format": "half_open_xyxy"},
        "scenes": scenes,
        "excluded_scenes": [
            {"scene_id": f"{subset}_excluded", "image_id": "excluded", "reason": "missing_or_unusable_render", "errors": ["render missing"]}
        ],
        "summary": {"total_scenes": len(scenes), "valid_scenes": len(scenes)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def write_renders(repo_root: Path, scene: Dict[str, Any]) -> None:
    render_dir = repo_root / scene["render_dir"]
    render_dir.mkdir(parents=True, exist_ok=True)
    files = ["full.png", "layer_0.png", "progressive_strip.png"]
    files += [f"layer_{layer['layer_id']}.png" for layer in scene["layers"]]
    for name in files:
        (render_dir / name).write_bytes(PNG_1X1)


def write_selected_renders(repo_root: Path, scene: Dict[str, Any]) -> None:
    """Copy the render stack into data/selected_1200/<subset>/<image_id>/ so the
    scene is part of the curated selection and its (repointed) images exist."""
    dest = repo_root / "data" / "selected_1200" / scene["subset"] / scene["image_id"]
    dest.mkdir(parents=True, exist_ok=True)
    files = ["full.png", "layer_0.png", "progressive_strip.png"]
    files += [f"layer_{layer['layer_id']}.png" for layer in scene["layers"]]
    for name in files:
        (dest / name).write_bytes(PNG_1X1)


@pytest.fixture()
def fake_repo(tmp_path: Path) -> RepoPaths:
    """A minimal repo with 4 LVIS + 3 LAION scenes and real (1x1) PNGs.

    Every scene is also placed under data/selected_1200/, so the whole fixture
    set is the reviewable selection.
    """
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)

    lvis_scenes = [
        make_scene("lvis", "000000000001", n_layers=2),
        make_scene("lvis", "000000000002", n_layers=3, with_occlusion=True),
        make_scene("lvis", "000000000003", n_layers=1),
        make_scene("lvis", "000000000004", n_layers=2, layer_without_caption=2),
    ]
    laion_scenes = [
        make_scene("laion", "000000101", n_layers=2),
        make_scene("laion", "000000102", n_layers=1),
        make_scene("laion", "000000103", n_layers=4, with_occlusion=True),
    ]
    write_source_file(repo / "data/derived/mulan-scenes/lvis_scenes.json", "lvis", lvis_scenes)
    write_source_file(repo / "data/derived/mulan-scenes/laion_scenes.json", "laion", laion_scenes)
    for scene in lvis_scenes + laion_scenes:
        write_renders(repo, scene)
        write_selected_renders(repo, scene)

    return RepoPaths(repo)


@pytest.fixture()
def service(fake_repo: RepoPaths) -> ReviewService:
    svc = ReviewService(fake_repo, seed=20260719)
    svc.rebuild_index()
    svc.set_active_reviewer("r1")
    return svc


def make_valid_review(scene: Dict[str, Any], exclude: Optional[List[int]] = None, primary: str = "direct_ask") -> Dict[str, Any]:
    """A review payload that passes acceptance validation."""
    exclude = exclude or []
    layer_reviews: Dict[str, Any] = {}
    for layer in scene["layers"]:
        lid = layer["layer_id"]
        if lid in exclude:
            layer_reviews[str(lid)] = {"decision": "exclude", "exclusion_reason": "duplicate"}
        else:
            layer_reviews[str(lid)] = {
                "decision": "keep",
                "label": f"object {lid}",
                "caption": f"reviewed caption for layer {lid}",
                "geometry_status": "correct",
            }
    suitable = [primary]
    return {
        "overall_quality": 4,
        "layer_reviews": layer_reviews,
        "reviewed_image_caption": "a reviewed full-image caption",
        "reviewed_image_caption_llava": "a reviewed LLaVA full-image caption",
        "reviewed_background_caption": "a reviewed background caption",
        "reviewed_background_caption_llava": "a reviewed LLaVA background caption",
        "suitable_categories": suitable,
        "primary_category": primary,
        "category_decision_made": True,
        "human_questions": {primary: "What object is visible in the scene?"},
        "human_answers": {primary: "An object is visible."},
        "duration_seconds": 42,
    }
