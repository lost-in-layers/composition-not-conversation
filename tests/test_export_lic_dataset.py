import json
import tempfile
import unittest
from pathlib import Path

from src.vecgen.export_lic_dataset import (
    REPO_ROOT,
    ExportError,
    aligned_schedule,
    build_sample,
    casual_opener,
    demand_level,
    dependency_order,
    export,
    index_scene,
    misaligned_schedule,
    oracle_layer_ids,
    visual_reveal_order,
)


def make_scene() -> dict:
    return {
        "scene_id": "lvis_0001",
        "subset": "lvis",
        "full_path": "data/selected_1200/lvis/0001/full.png",
        "background": {
            "layer_id": 0,
            "path": "data/selected_1200/lvis/0001/layer_0.png",
        },
        "layers": [
            {
                "layer_id": 1,
                "path": "data/selected_1200/lvis/0001/layer_1.png",
                "label": "white cat",
            },
            {
                "layer_id": 2,
                "path": "data/selected_1200/lvis/0001/layer_2.png",
                "label": "bulb",
            },
        ],
        "questions": [
            {
                "question_id": "lvis_0001::direct_ask",
                "category": "direct_ask",
                "question": "What is the color of the bulb?",
                "answer": "Blue",
                "confidence": "high",
                "shards": [
                    {"shard_id": 1, "role": "intent", "text": "Determine a property of a particular object in the image."},
                    {"shard_id": 2, "role": "answer_type", "text": "The requested property is its color."},
                    {"shard_id": 3, "role": "target", "text": "The target object is the bulb."},
                ],
                "gold_layer_ids": [2],
                "distractor_layer_ids": [0, 1],
                "layer_roles": {"2": "attribute_source"},
                "minimal_sufficient_layer_sets": [[2]],
                "question_shard_support": {"1": [], "2": [2], "3": [2]},
                "evidence_dependencies": [],
            }
        ],
    }


class DependencyOrderTests(unittest.TestCase):
    def test_before_constraints_are_respected(self) -> None:
        dependencies = [
            {"after": 0, "before": 4, "reason": "reference_resolution"},
            {"after": 5, "before": 0, "reason": "relation_resolution"},
        ]
        self.assertEqual(dependency_order([4, 0, 5], dependencies), [4, 0, 5])
        self.assertEqual(dependency_order([5, 0, 4], dependencies), [4, 0, 5])

    def test_cycles_fall_back_to_annotated_order(self) -> None:
        dependencies = [
            {"before": 1, "after": 2},
            {"before": 2, "after": 1},
        ]
        self.assertEqual(dependency_order([1, 2], dependencies), [1, 2])

    def test_dependencies_outside_the_set_are_ignored(self) -> None:
        self.assertEqual(
            dependency_order([1, 2], [{"before": 9, "after": 2}]), [1, 2]
        )


class OpenerTests(unittest.TestCase):
    def test_annotator_voice_becomes_user_voice(self) -> None:
        self.assertEqual(
            casual_opener("Determine a property of a particular object in the image."),
            "can you help me figure out a property of a particular object in this image?",
        )

    def test_unknown_lead_verb_is_left_alone(self) -> None:
        self.assertEqual(
            casual_opener("Work out how many cats there are."),
            "can you help me work out how many cats there are?",
        )

    def test_empty_text_gets_a_usable_opener(self) -> None:
        self.assertTrue(casual_opener("   ").endswith("?"))


class DemandLevelTests(unittest.TestCase):
    def test_single_layer_without_distractors_is_level_one(self) -> None:
        self.assertEqual(
            demand_level({"gold_layer_ids": [2], "distractor_layer_ids": []}), 1
        )

    def test_single_layer_with_distractors_is_level_two(self) -> None:
        self.assertEqual(
            demand_level({"gold_layer_ids": [2], "distractor_layer_ids": [0, 1]}), 2
        )

    def test_multi_hop_reference_is_level_four(self) -> None:
        self.assertEqual(
            demand_level(
                {
                    "gold_layer_ids": [4, 0],
                    "distractor_layer_ids": [1],
                    "evidence_dependencies": [{"before": 4, "after": 0}],
                }
            ),
            4,
        )

    def test_many_gold_layers_reach_level_five(self) -> None:
        self.assertEqual(
            demand_level(
                {"gold_layer_ids": [0, 1, 2, 3], "distractor_layer_ids": [4]}
            ),
            5,
        )


class SceneIndexTests(unittest.TestCase):
    def test_background_and_instances_are_both_indexed(self) -> None:
        index = index_scene(make_scene())
        self.assertEqual(index.ordered_layer_ids, [0, 1, 2])
        self.assertTrue(index.layers[0].is_background)
        self.assertEqual(index.layers[0].display, "layer 0 (the background layer)")

    def test_prompt_labels_never_leak_the_object_name(self) -> None:
        # `label` is often the answer ("computer mouse"), so it must not reach
        # the prompt; only the layer number identifies a layer.
        index = index_scene(make_scene())
        self.assertEqual(index.layers[1].display, "layer 1")
        self.assertEqual(index.layers[2].display, "layer 2")
        self.assertNotIn("bulb", index.layers[2].display)

    def test_unknown_layer_is_rejected(self) -> None:
        index = index_scene(make_scene())
        with self.assertRaises(ExportError):
            index.resolve([7])


class SampleBuilderMixin:
    def setUp(self) -> None:
        self.scene = make_scene()
        self.question = self.scene["questions"][0]
        # Scene paths resolve against REPO_ROOT, so the dataset directory has
        # to sit inside the same tree for the relative paths to be meaningful.
        self.render_dir = REPO_ROOT / "data/derived/lic_renders"
        self.base_dir = REPO_ROOT / "data/derived/lic"

    def build(self, **overrides):
        kwargs = dict(
            variant="text",
            layer_order="canonical",
            opener_style="casual",
            include_full_image=True,
            render_dir=self.render_dir,
            base_dir=self.base_dir,
            detail="high",
        )
        kwargs.update(overrides)
        return build_sample(self.scene, self.question, **kwargs)


class BuildSampleTests(SampleBuilderMixin, unittest.TestCase):

    def test_text_variant_carries_the_full_image_and_no_shard_images(self) -> None:
        sample = self.build()
        self.assertEqual(sample["task"], "layered_vqa")
        self.assertEqual(sample["task_id"], "lvis_0001::direct_ask")
        self.assertEqual(len(sample["images"]), 1)
        self.assertEqual(sample["images"][0]["label"], "the full image")
        self.assertTrue(all("images" not in s for s in sample["shards"]))

    def test_image_paths_are_relative_to_the_dataset_file(self) -> None:
        sample = self.build()
        # multimodal.image_base_dir resolves against the JSON file's parent.
        self.assertEqual(
            sample["images"][0]["path"],
            "../../selected_1200/lvis/0001/full.png",
        )

    def test_casual_opener_replaces_only_the_first_shard(self) -> None:
        sample = self.build()
        self.assertTrue(sample["shards"][0]["shard"].startswith("can you help me"))
        self.assertEqual(
            sample["shards"][1]["shard"], "The requested property is its color."
        )
        self.assertEqual(
            sample["meta"]["intent_shard_raw"],
            "Determine a property of a particular object in the image.",
        )

    def test_raw_opener_style_is_untouched(self) -> None:
        sample = self.build(opener_style="raw")
        self.assertEqual(
            sample["shards"][0]["shard"],
            "Determine a property of a particular object in the image.",
        )

    def test_aligned_vq_attaches_supporting_layers_once(self) -> None:
        sample = self.build(variant="aligned_vq", include_full_image=False)
        self.assertEqual(sample["images"], [])
        shards = sample["shards"]
        self.assertNotIn("images", shards[0])  # intent shard supports no layer
        self.assertEqual([i["label"] for i in shards[1]["images"]], ["layer 2"])
        # Layer 2 already arrived, so shard 3 adds nothing new.
        self.assertNotIn("images", shards[2])
        self.assertEqual(sample["meta"]["reveal_order"], [2])

    def test_aligned_vq_points_at_flattened_renders(self) -> None:
        sample = self.build(variant="aligned_vq", include_full_image=False)
        self.assertEqual(
            sample["shards"][1]["images"][0]["path"],
            "../lic_renders/lvis_0001/layer_2.png",
        )

    def test_misaligned_vq_moves_the_same_layers_to_other_turns(self) -> None:
        aligned = self.build(variant="aligned_vq", include_full_image=False)
        misaligned = self.build(variant="misaligned_vq", include_full_image=False)

        def per_turn(sample):
            return [len(s.get("images", [])) for s in sample["shards"]]

        # Information-matched: same layers arrive, on a different schedule.
        self.assertEqual(
            sorted(aligned["meta"]["reveal_order"]),
            sorted(misaligned["meta"]["reveal_order"]),
        )
        self.assertEqual(sum(per_turn(aligned)), sum(per_turn(misaligned)))
        self.assertNotEqual(per_turn(aligned), per_turn(misaligned))

        # Layer 2 supports shards 2 and 3, so it may only travel with shard 1.
        support = {
            int(k): set(v)
            for k, v in misaligned["meta"]["question_shard_support"].items()
        }
        for shard in misaligned["shards"]:
            for image in shard.get("images", []):
                layer_id = int(image["path"].rsplit("layer_", 1)[1].split(".")[0])
                self.assertNotIn(layer_id, support[shard["shard_id"]])
        self.assertEqual(misaligned["meta"]["misaligned_violations"], [])
        self.assertEqual(
            misaligned["meta"]["misaligned_schedule"],
            {"1": [2], "2": [], "3": []},
        )
        self.assertIsNone(aligned["meta"]["misaligned_schedule"])

    def test_every_variant_carries_the_full_rgb_as_recap_images(self) -> None:
        for variant in ("text", "aligned_vq", "misaligned_vq", "visual"):
            sample = self.build(variant=variant, include_full_image=variant == "text")
            self.assertEqual(len(sample["recap_images"]), 1, variant)
            self.assertEqual(sample["recap_images"][0]["label"], "the full image")
            self.assertIn("full.png", sample["recap_images"][0]["path"])

    def test_visual_variant_opens_with_the_whole_question(self) -> None:
        sample = self.build(variant="visual", include_full_image=False)
        self.assertEqual(sample["shards"][0]["shard"], "What is the color of the bulb?")
        self.assertEqual(len(sample["shards"]), 4)  # question + three layers
        self.assertEqual(sample["meta"]["reveal_order"], [0, 1, 2])

    def test_visual_evidence_first_and_last_reverse_the_schedule(self) -> None:
        first = self.build(
            variant="visual", layer_order="evidence_first", include_full_image=False
        )
        last = self.build(
            variant="visual", layer_order="evidence_last", include_full_image=False
        )
        self.assertEqual(first["meta"]["reveal_order"][0], 2)  # gold layer leads
        self.assertEqual(last["meta"]["reveal_order"][-1], 2)  # gold layer trails

    def test_shard_ids_stay_contiguous_in_the_visual_variant(self) -> None:
        sample = self.build(variant="visual", include_full_image=False)
        self.assertEqual(
            [s["shard_id"] for s in sample["shards"]], [1, 2, 3, 4]
        )

    def test_meta_carries_the_grounding_labels(self) -> None:
        meta = self.build()["meta"]
        self.assertEqual(meta["gold_layer_ids"], [2])
        self.assertEqual(meta["distractor_layer_ids"], [0, 1])
        self.assertEqual(meta["minimal_sufficient_layer_sets"], [[2]])
        self.assertEqual(meta["shard_roles"], {"1": "intent", "2": "answer_type", "3": "target"})
        self.assertEqual(meta["demand_level"], 2)

    def test_every_variant_names_the_candidate_layers_and_the_background(self) -> None:
        for variant in ("text", "visual", "grounding", "oracle_gold"):
            meta = self.build(variant=variant, include_full_image=False)["meta"]
            self.assertEqual(meta["candidate_layer_ids"], [0, 1, 2], variant)
            self.assertEqual(meta["background_layer_id"], 0, variant)


class SingleTurnVariantTests(SampleBuilderMixin, unittest.TestCase):
    """Experiment A: one turn, every image attached at sample level."""

    def test_grounding_sends_the_full_image_and_every_retained_layer(self) -> None:
        sample = self.build(variant="grounding", include_full_image=True)
        self.assertEqual(
            [image["label"] for image in sample["images"]],
            [
                "the full image",
                "layer 0 (the background layer)",
                "layer 1",
                "layer 2",
            ],
        )
        # One turn holding the whole question, no per-shard images.
        self.assertEqual(len(sample["shards"]), 1)
        self.assertEqual(sample["shards"][0]["shard"], "What is the color of the bulb?")
        self.assertNotIn("images", sample["shards"][0])
        self.assertEqual(sample["meta"]["reveal_order"], [0, 1, 2])

    def test_grounding_asks_for_layer_selection(self) -> None:
        self.assertEqual(
            self.build(variant="grounding")["meta"]["response_format"],
            "answer_and_layers",
        )
        # Every other variant asks for the answer alone.
        for variant in ("text", "visual", "aligned_vq", "oracle_gold"):
            self.assertEqual(
                self.build(variant=variant, include_full_image=False)["meta"]["response_format"],
                "answer",
                variant,
            )

    def test_oracle_sends_only_the_minimal_sufficient_set(self) -> None:
        sample = self.build(variant="oracle_gold", include_full_image=False)
        self.assertEqual([image["label"] for image in sample["images"]], ["layer 2"])
        self.assertEqual(
            sample["images"][0]["path"], "../lic_renders/lvis_0001/layer_2.png"
        )
        self.assertEqual(sample["meta"]["oracle_layer_ids"], [2])
        self.assertEqual(sample["meta"]["oracle_source"], "minimal_sufficient")
        self.assertEqual(len(sample["shards"]), 1)

    def test_oracle_ids_are_null_on_the_other_variants(self) -> None:
        for variant in ("text", "grounding"):
            meta = self.build(variant=variant)["meta"]
            self.assertIsNone(meta["oracle_layer_ids"], variant)
            self.assertIsNone(meta["oracle_source"], variant)


class OracleLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = index_scene(make_scene())

    def question(self, **overrides) -> dict:
        question = {
            "question_id": "q1",
            "minimal_sufficient_layer_sets": [[2]],
            "gold_layer_ids": [1, 2],
            "evidence_dependencies": [],
        }
        question.update(overrides)
        return question

    def test_minimal_set_wins_over_the_full_gold_set(self) -> None:
        self.assertEqual(
            oracle_layer_ids(self.question(), self.index), ([2], "minimal_sufficient")
        )

    def test_gold_is_the_fallback_when_no_minimal_set_is_annotated(self) -> None:
        self.assertEqual(
            oracle_layer_ids(self.question(minimal_sufficient_layer_sets=[]), self.index),
            ([1, 2], "gold_layer_ids"),
        )

    def test_a_minimal_set_naming_an_excluded_layer_is_skipped(self) -> None:
        question = self.question(minimal_sufficient_layer_sets=[[7], [1]])
        self.assertEqual(oracle_layer_ids(question, self.index), ([1], "minimal_sufficient"))

    def test_dependencies_order_the_delivered_set(self) -> None:
        question = self.question(
            minimal_sufficient_layer_sets=[[1, 2]],
            evidence_dependencies=[{"before": 2, "after": 1}],
        )
        self.assertEqual(oracle_layer_ids(question, self.index)[0], [2, 1])

    def test_a_question_with_no_usable_evidence_is_skipped(self) -> None:
        question = self.question(minimal_sufficient_layer_sets=[[7]], gold_layer_ids=[7])
        with self.assertRaises(ExportError):
            oracle_layer_ids(question, self.index)


class ScheduleTests(unittest.TestCase):
    def test_a_layer_supporting_two_shards_arrives_once(self) -> None:
        shards = [{"shard_id": 1}, {"shard_id": 2}, {"shard_id": 3}]
        schedule = aligned_schedule(shards, {1: [], 2: [2], 3: [2]})
        self.assertEqual(schedule, {1: [], 2: [2], 3: []})

    def test_misaligned_layers_never_land_on_a_supporting_shard(self) -> None:
        # Layer 2 supports shards 2 AND 3: the old rotate-by-one would have
        # parked it on shard 3, which still supports it. The misalignment has
        # to be checked against the support map, not the aligned schedule.
        aligned = {1: [], 2: [2], 3: [], 4: [4, 5]}
        support = {1: [], 2: [2], 3: [2], 4: [4, 5]}
        schedule, violations = misaligned_schedule(aligned, support)

        self.assertEqual(violations, [])
        # Same layers overall, same group sizes, just on other turns.
        self.assertEqual(
            sorted(x for v in schedule.values() for x in v),
            sorted(x for v in aligned.values() for x in v),
        )
        self.assertEqual(
            sorted(len(v) for v in schedule.values()),
            sorted(len(v) for v in aligned.values()),
        )
        for shard_id, group in schedule.items():
            for layer_id in group:
                self.assertNotIn(layer_id, support[shard_id])

    def test_impossible_misalignment_is_minimized_and_reported(self) -> None:
        # Every shard supports layer 2, so it must land somewhere it is
        # supported; the residual pairing is returned for meta.
        aligned = {1: [2], 2: [], 3: []}
        support = {1: [2], 2: [2], 3: [2]}
        schedule, violations = misaligned_schedule(aligned, support)
        self.assertEqual(sorted(x for v in schedule.values() for x in v), [2])
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0][1], 2)

    def test_misaligned_is_a_no_op_without_layer_groups(self) -> None:
        self.assertEqual(
            misaligned_schedule({1: [], 2: []}, {1: [], 2: []}),
            ({1: [], 2: []}, []),
        )


class VisualRevealOrderTests(unittest.TestCase):
    def test_canonical_order_reveals_every_retained_layer(self) -> None:
        scene = make_scene()
        index = index_scene(scene)
        order = visual_reveal_order(scene["questions"][0], index, "canonical")
        self.assertEqual(order, [0, 1, 2])


class ExportTests(unittest.TestCase):
    def test_questions_referencing_excluded_layers_are_skipped_not_dropped_silently(
        self,
    ) -> None:
        scene = make_scene()
        scene["questions"][0]["gold_layer_ids"] = [9]
        scene["questions"][0]["question_shard_support"] = {"1": [], "2": [9], "3": [9]}
        with tempfile.TemporaryDirectory() as tmp:
            samples, skipped = export(
                [scene],
                variant="aligned_vq",
                layer_order="canonical",
                opener_style="casual",
                include_full_image=False,
                render_dir=Path(tmp) / "renders",
                output_path=Path(tmp) / "out.json",
                detail="high",
            )
        self.assertEqual(samples, [])
        self.assertEqual(len(skipped), 1)
        self.assertIn("layer 9", skipped[0])

    def test_category_filter_and_limit_apply(self) -> None:
        scene = make_scene()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.json"
            samples, _ = export(
                [scene],
                variant="text",
                layer_order="canonical",
                opener_style="casual",
                include_full_image=True,
                render_dir=Path(tmp) / "renders",
                output_path=output,
                detail="high",
                categories={"counting"},
            )
            self.assertEqual(samples, [])

            samples, _ = export(
                [scene],
                variant="text",
                layer_order="canonical",
                opener_style="casual",
                include_full_image=True,
                render_dir=Path(tmp) / "renders",
                output_path=output,
                detail="high",
                categories={"direct_ask"},
            )
            self.assertEqual(len(samples), 1)
            json.dumps(samples)  # the export must stay JSON-serialisable


if __name__ == "__main__":
    unittest.main()
