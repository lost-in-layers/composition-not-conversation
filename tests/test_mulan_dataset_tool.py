import csv
import hashlib
import json
import pickle
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from src.vecgen.mulan_dataset_tool import (
    INVENTORY_FIELDS,
    command_verify,
    inspect_pzl,
    native_value_manifest,
    render_one,
    verify_inventory_row,
)


class MulanDatasetToolTests(unittest.TestCase):
    def test_verify_rejects_an_empty_filtered_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inventory_path = Path(directory) / "inventory.csv"
            with inventory_path.open(
                "w", encoding="utf-8-sig", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=INVENTORY_FIELDS)
                writer.writeheader()
                writer.writerow({
                    "image_id": "1",
                    "object_layers": "1",
                    "render_status": "missing",
                })

            args = SimpleNamespace(
                inventory=str(inventory_path),
                min_layers=0,
                max_layers=0,
                render_status="existing",
                limit=0,
                workers=1,
                require_renders=True,
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "No inventory rows matched",
            ):
                command_verify(args)

    def test_manifest_preserves_text_and_describes_arrays(self) -> None:
        array = np.arange(12, dtype=np.uint8).reshape(3, 4)
        manifest = native_value_manifest({"caption": "red bicycle", "mask": array})

        self.assertEqual(manifest["caption"], "red bicycle")
        self.assertEqual(manifest["mask"]["__native_type__"], "numpy.ndarray")
        self.assertEqual(manifest["mask"]["dtype"], "uint8")
        self.assertEqual(manifest["mask"]["shape"], [3, 4])
        self.assertEqual(manifest["mask"]["nbytes"], 12)
        self.assertEqual(
            manifest["mask"]["sha256"],
            hashlib.sha256(array.tobytes()).hexdigest(),
        )

    def test_inspect_and_verify_lossless_annotation_reference(self) -> None:
        annotation = {
            "captioning": {"blip2": {"model": "test-model"}},
            "image": {"blip2": "a person with a bag", "llava": "detailed"},
            "background": {
                "blip2": "a road",
                "llava": "a road behind the subject",
                "original_image_mask": np.ones((2, 3, 1), dtype=np.uint8),
                "inpainted_delta": np.zeros((2, 3, 3), dtype=np.int16),
            },
            "instances": {
                1: {
                    "blip2": "a red bag",
                    "original_image_mask": np.ones((2, 3, 1), dtype=np.uint8),
                    "instance_mask": np.ones((2, 3, 1), dtype=np.uint8),
                    "inpainted_delta": np.zeros((2, 3, 3), dtype=np.int16),
                    "instance_alpha": np.full((2, 3), 255, dtype=np.uint8),
                }
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            pzl_path = Path(directory) / "000000000001.p.zl"
            compressed = zlib.compress(pickle.dumps(annotation))
            pzl_path.write_bytes(compressed)

            inspected = inspect_pzl(str(pzl_path))
            self.assertEqual(inspected["image_blip2"], "a person with a bag")
            self.assertEqual(inspected["background_blip2"], "a road")
            self.assertEqual(
                json.loads(inspected["instance_blip2_json"]),
                {"1": "a red bag"},
            )

            metadata = json.loads(inspected["annotation_metadata_json"])
            self.assertEqual(metadata["instances"]["1"]["blip2"], "a red bag")
            self.assertEqual(
                metadata["instances"]["1"]["instance_alpha"]["shape"],
                [2, 3],
            )

            row = {
                "image_id": inspected["image_id"],
                "pzl_path": str(pzl_path),
                "annotation_sha256": inspected["annotation_sha256"],
                "annotation_size_bytes": str(inspected["annotation_size_bytes"]),
                "object_layers": "1",
            }
            image_id, error = verify_inventory_row(row, directory, False)
            self.assertEqual(image_id, "000000000001")
            self.assertEqual(error, "")

            output_root = Path(directory) / "renders"
            instance_dir = output_root / "lvis" / "000000000001"
            instance_dir.mkdir(parents=True)
            Image.new("RGB", (3, 2)).save(instance_dir / "full.png")
            Image.new("RGB", (3, 2)).save(instance_dir / "layer_0.png")
            Image.new("RGBA", (3, 2)).save(instance_dir / "layer_1.png")

            render_result = render_one(
                {
                    "image_id": "000000000001",
                    "subset": "lvis",
                    "pzl_path": str(pzl_path),
                    "image_ref": "",
                },
                directory,
                str(output_root),
                1,
                False,
                True,
            )
            self.assertEqual(render_result["render_status"], "existing")


if __name__ == "__main__":
    unittest.main()
