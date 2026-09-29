"""Invariants the shipped `data/derived/lic/*.json` exports must satisfy.

These check the files that actually run, not the exporter that produced them.
`layered_vqa_misaligned_vq.json` shipped for a while with 15 samples whose gold
layers were replaced by the background layer, which the exporter unit tests
could not see because the file predates the fixed exporter. Skipped when the
exports are not in the checkout.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

LIC_DIR = Path(__file__).resolve().parents[1] / "data" / "derived" / "lic"
LAYER_RE = re.compile(r"layer_(\d+)\.png$")


def load(name: str) -> list[dict]:
    path = LIC_DIR / name
    if not path.exists():
        raise unittest.SkipTest(f"{path} not in this checkout")
    return json.loads(path.read_text(encoding="utf-8"))


def shard_layer_ids(sample: dict) -> list[list[int]]:
    """Per shard, the layer ids it attaches — in shard order."""
    out = []
    for shard in sample["shards"]:
        ids = []
        for image in shard.get("images") or []:
            match = LAYER_RE.search(image["path"])
            if match:
                ids.append(int(match.group(1)))
        out.append(ids)
    return out


def delivered(sample: dict) -> set[int]:
    return {lid for ids in shard_layer_ids(sample) for lid in ids}


class TestLayerCarryingExports(unittest.TestCase):
    NAMES = (
        "layered_vqa_aligned_vq.json",
        "layered_vqa_misaligned_vq.json",
        "layered_vqa_visual_canonical.json",
        "layered_vqa_visual_evidence_first.json",
        "layered_vqa_visual_evidence_last.json",
    )

    def test_every_gold_layer_is_delivered(self) -> None:
        for name in self.NAMES:
            for sample in load(name):
                gold = {int(x) for x in sample["meta"]["gold_layer_ids"]}
                available = delivered(sample) | {
                    int(match.group(1))
                    for image in sample.get("images") or []
                    if (match := LAYER_RE.search(image["path"]))
                }
                missing = gold - available
                self.assertFalse(
                    missing,
                    f"{name} {sample['task_id']}: gold layers {sorted(missing)} "
                    "are never delivered",
                )

    def test_referenced_renders_exist(self) -> None:
        for name in self.NAMES:
            for sample in load(name):
                for shard in sample["shards"]:
                    for image in shard.get("images") or []:
                        path = (LIC_DIR / image["path"]).resolve()
                        self.assertTrue(path.exists(), f"{name}: missing {path}")


class TestMisalignedDelivery(unittest.TestCase):
    """MISALIGNED_VQ pairs each shard with layers that do not support it.

    Every shard that carries images must carry non-supporting ones, and the gold
    layers must still reach the model somewhere in the conversation. On the 15
    questions where every shard supports the gold group, both cannot hold for the
    same turn, so the gold group is added on one extra, late shard and the sample
    records that in `meta.misaligned_gold_makeup`. Those are the only shards
    allowed to carry a layer they support.
    """

    def setUp(self) -> None:
        self.aligned = {s["task_id"]: s for s in load("layered_vqa_aligned_vq.json")}
        self.misaligned = load("layered_vqa_misaligned_vq.json")

    def test_same_task_ids(self) -> None:
        self.assertEqual(
            sorted(self.aligned), sorted(s["task_id"] for s in self.misaligned)
        )

    def test_delivers_at_least_what_aligned_delivers(self) -> None:
        """Misaligned may not deliver *less* evidence than the arm it is compared to."""
        for sample in self.misaligned:
            twin = self.aligned[sample["task_id"]]
            self.assertLessEqual(
                delivered(twin),
                delivered(sample),
                f"{sample['task_id']}: misaligned is missing layers aligned delivers",
            )

    def test_untouched_samples_stay_information_matched(self) -> None:
        """Without a make-up delivery the two arms must match exactly.

        Same layer set and same per-turn image counts is what makes the contrast
        isolate alignment rather than how much evidence arrived.
        """
        for sample in self.misaligned:
            if "misaligned_gold_makeup" in sample["meta"]:
                continue
            twin = self.aligned[sample["task_id"]]
            self.assertEqual(
                delivered(sample), delivered(twin), f"{sample['task_id']}: layer set"
            )
            self.assertEqual(
                sorted(len(ids) for ids in shard_layer_ids(sample)),
                sorted(len(ids) for ids in shard_layer_ids(twin)),
                f"{sample['task_id']}: per-turn image counts",
            )

    def test_schedule_matches_the_attached_images(self) -> None:
        for sample in self.misaligned:
            schedule = sample["meta"]["misaligned_schedule"]
            self.assertIsNotNone(schedule, sample["task_id"])
            attached = {
                str(shard["shard_id"]): ids
                for shard, ids in zip(sample["shards"], shard_layer_ids(sample))
            }
            self.assertEqual(
                {k: v for k, v in schedule.items() if v},
                {k: v for k, v in attached.items() if v},
                f"{sample['task_id']}: meta.misaligned_schedule vs shard images",
            )

    def test_no_shard_carries_a_layer_it_supports(self) -> None:
        """Except the recorded make-up shard, where no alternative exists."""
        for sample in self.misaligned:
            meta = sample["meta"]
            support = {
                int(k): {int(x) for x in v}
                for k, v in meta["question_shard_support"].items()
            }
            makeup = meta.get("misaligned_gold_makeup") or {}
            exempt_shard = makeup.get("added_shard_id")
            exempt_layers = set(makeup.get("added_layer_ids") or [])
            for shard, layer_ids in zip(sample["shards"], shard_layer_ids(sample)):
                shard_id = int(shard["shard_id"])
                for layer_id in layer_ids:
                    if layer_id not in support.get(shard_id, set()):
                        continue
                    self.assertTrue(
                        shard_id == exempt_shard and layer_id in exempt_layers,
                        f"{sample['task_id']}: shard {shard_id} carries layer "
                        f"{layer_id}, which it supports, and is not the recorded "
                        "make-up delivery",
                    )

    def test_violations_are_empty_so_every_sample_is_analysable(self) -> None:
        for sample in self.misaligned:
            self.assertFalse(
                sample["meta"].get("misaligned_violations"),
                f"{sample['task_id']}: a non-empty misaligned_violations list "
                "drops this sample from the aligned-vs-misaligned contrast",
            )

    def test_no_stale_background_substitution_marker(self) -> None:
        for sample in self.misaligned:
            self.assertNotIn("misaligned_fallback", sample["meta"], sample["task_id"])


if __name__ == "__main__":
    unittest.main()
