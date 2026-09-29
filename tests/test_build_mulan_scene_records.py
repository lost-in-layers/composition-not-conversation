import argparse
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.vecgen.build_mulan_scene_records import (
    build_collection,
    connected_components_4,
    mask_geometry,
)
from src.vecgen.mulan_dataset_tool import INVENTORY_FIELDS


class BuildMulanSceneRecordsTests(unittest.TestCase):
    def test_mask_geometry_and_components(self) -> None:
        alpha = np.zeros((4, 5), dtype=np.uint8)
        alpha[0:2, 1:3] = 255
        alpha[3, 4] = 255

        geometry, support = mask_geometry(alpha, 1, 128)

        self.assertEqual(geometry["bbox_xyxy"], [1, 0, 5, 4])
        self.assertEqual(geometry["support_area_pixels"], 5)
        self.assertEqual(geometry["core_area_pixels"], 5)
        self.assertEqual(geometry["connected_components_4"], 2)
        self.assertTrue(geometry["touches_image_border"])
        self.assertEqual(connected_components_4(support), 2)

        separated_rows = np.array([[True], [False], [True]])
        self.assertEqual(connected_components_4(separated_rows), 2)

    def test_builds_pretty_json_with_alpha_masks_and_numeric_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory_root = root / "data" / "pre-data" / "inventories"
            render_dir = root / "data" / "pre-data" / "renders" / "lvis" / "lvis" / "7"
            annotation_dir = root / "data" / "pre-data" / "annotations"
            output_dir = root / "data" / "derived" / "mulan-scenes"
            inventory_root.mkdir(parents=True)
            render_dir.mkdir(parents=True)
            annotation_dir.mkdir(parents=True)

            background = Image.new("RGB", (6, 4), (10, 20, 30))
            background.save(render_dir / "layer_0.png")

            alpha_two = np.zeros((4, 6), dtype=np.uint8)
            alpha_two[1:3, 1:3] = 255
            layer_two = Image.new("RGBA", (6, 4), (200, 0, 0, 0))
            layer_two.putalpha(Image.fromarray(alpha_two))
            layer_two.save(render_dir / "layer_2.png")

            alpha_ten = np.zeros((4, 6), dtype=np.uint8)
            alpha_ten[2:4, 2:5] = 255
            layer_ten = Image.new("RGBA", (6, 4), (0, 200, 0, 0))
            layer_ten.putalpha(Image.fromarray(alpha_ten))
            layer_ten.save(render_dir / "layer_10.png")

            canvas = background.convert("RGBA")
            canvas = Image.alpha_composite(canvas, layer_two)
            canvas = Image.alpha_composite(canvas, layer_ten)
            canvas.convert("RGB").save(render_dir / "full.png")

            pzl_path = annotation_dir / "7.p.zl"
            pzl_path.write_bytes(b"trusted-test-payload")
            manifest = {
                "instances": {
                    "2": {
                        "blip2": "red square",
                        "instance_alpha": {
                            "mode": "L",
                            "dtype": "uint8",
                            "shape": [4, 6],
                            "sha256": hashlib.sha256(alpha_two.tobytes()).hexdigest(),
                        },
                    },
                    "10": {
                        "blip2": "green rectangle",
                        "instance_alpha": {
                            "mode": "L",
                            "dtype": "uint8",
                            "shape": [4, 6],
                            "sha256": hashlib.sha256(alpha_ten.tobytes()).hexdigest(),
                        },
                    },
                }
            }
            row = {field: "" for field in INVENTORY_FIELDS}
            row.update({
                "subset": "lvis",
                "image_id": "7",
                "pzl_path": str(pzl_path),
                "annotation_sha256": hashlib.sha256(pzl_path.read_bytes()).hexdigest(),
                "annotation_size_bytes": str(pzl_path.stat().st_size),
                "annotation_captioning_json": json.dumps({"blip2": {"hf_id": "test"}}),
                "annotation_metadata_json": json.dumps(manifest),
                "image_blip2": "two colored shapes",
                "image_llava": "A red square and green rectangle.",
                "background_blip2": "dark background",
                "background_llava": "A dark background.",
                "instance_keys_json": json.dumps(["2", "10"]),
                "instance_blip2_json": json.dumps({
                    "2": "red square",
                    "10": "green rectangle",
                }),
                "categories": "square|rectangle",
                "object_layers": "2",
                "total_layers_including_background": "3",
                "render_status": "existing",
                "render_dir": str(render_dir),
                "full_path": str(render_dir / "full.png"),
                "layer_paths": json.dumps([
                    str(render_dir / "layer_0.png"),
                    str(render_dir / "layer_10.png"),
                    str(render_dir / "layer_2.png"),
                ]),
            })
            inventory_path = inventory_root / "lvis.csv"
            with inventory_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=INVENTORY_FIELDS)
                writer.writeheader()
                writer.writerow(row)
                missing_row = dict(row)
                missing_row.update({
                    "image_id": "8",
                    "render_status": "missing",
                    "render_dir": "",
                    "full_path": "",
                    "layer_paths": "",
                })
                writer.writerow(missing_row)

            args = argparse.Namespace(
                overwrite=False,
                workers=2,
                support_threshold=1,
                core_threshold=128,
                near_threshold=0.1,
                containment_threshold=0.98,
                occlusion_min_overlap=0.01,
                skip_composite_validation=False,
                exclude_invalid=True,
            )
            output_path = output_dir / "lvis_scenes.json"
            summary = build_collection(
                "lvis",
                inventory_path,
                output_path,
                root,
                args,
            )

            parsed = json.loads(output_path.read_text(encoding="utf-8"))
            scene = parsed["scenes"][0]
            self.assertEqual(summary["valid_scenes"], 1)
            self.assertEqual(summary["source_inventory_scenes"], 2)
            self.assertEqual(summary["total_scenes"], 1)
            self.assertEqual(summary["invalid_scenes"], 0)
            self.assertEqual(summary["excluded_invalid_scenes"], 1)
            self.assertEqual(len(parsed["scenes"]), 1)
            self.assertEqual(parsed["excluded_scenes"][0]["scene_id"], "lvis_8")
            self.assertEqual(
                parsed["excluded_scenes"][0]["reason"],
                "missing_or_unusable_render",
            )
            self.assertEqual([layer["layer_id"] for layer in scene["layers"]], [2, 10])
            self.assertEqual([layer["z_index"] for layer in scene["layers"]], [1, 2])
            self.assertEqual(scene["layers"][0]["mask"]["encoding"], "rgba_alpha")
            self.assertNotIn("counts", scene["layers"][0]["mask"])
            self.assertTrue(scene["validation"]["composite"]["matches_full_png"])
            self.assertTrue(scene["annotation"]["matches_inventory_sha256"])
            self.assertEqual(scene["captions"]["blip2"], "two colored shapes")
            self.assertIn(
                {
                    "subject_layer_id": 10,
                    "predicate": "possible_occludes",
                    "object_layer_id": 2,
                    "score": 0.25,
                    "source": "mask_overlap_and_stack_order",
                },
                scene["relations"],
            )
            self.assertIn("\n    {\n      \"scene_id\"", output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
