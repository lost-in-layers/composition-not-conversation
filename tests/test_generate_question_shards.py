import json
import tempfile
import unittest
from pathlib import Path

from src.vecgen.generate_question_shards import (
    CATEGORIES,
    REPO_ROOT,
    AtomicUnit,
    EvidenceDependency,
    EvidencePlan,
    QuestionShard,
    load_questions,
    parse_categories,
    read_prompt_bundle,
    select_temp_records,
    shardability_label,
    validate_evidence,
    validate_shards,
)


class GenerateQuestionShardsTests(unittest.TestCase):
    @staticmethod
    def scene(questions: list[dict[str, str]]) -> dict:
        return {
            "scene_id": "scene_1",
            "full_path": "full.png",
            "width": 100,
            "height": 100,
            "background": {
                "path": "layer_0.png",
                "caption": "A plain background.",
            },
            "layers": [
                {
                    "layer_id": 1,
                    "path": "layer_1.png",
                    "caption": "An orange cat.",
                }
            ],
            "questions": questions,
        }

    def test_load_questions_flattens_records_with_stable_ids(self) -> None:
        payload = {
            "count": 1,
            "scenes": [
                self.scene(
                    [
                        {
                            "category": "direct_ask",
                            "question": "What color is the cat?",
                            "answer": "Orange",
                        },
                        {
                            "category": "counting",
                            "question": "How many cats are visible?",
                            "answer": "One",
                        },
                    ]
                )
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "accepted.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded, records = load_questions(path)

        self.assertEqual(loaded["count"], 1)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["question_id"], "scene_1::direct_ask")
        self.assertEqual(records[1]["question_id"], "scene_1::counting")

    def test_load_questions_rejects_duplicate_scene_categories(self) -> None:
        question = {
            "category": "direct_ask",
            "question": "What color is the cat?",
            "answer": "Orange",
        }
        payload = {
            "count": 1,
            "scenes": [self.scene([question, question])],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "accepted.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate question id"):
                load_questions(path)

    def test_validate_plan_accepts_complete_three_shard_short_question(self) -> None:
        units = [
            AtomicUnit(unit_id=1, source_span="What", function="intent"),
            AtomicUnit(unit_id=2, source_span="color", function="answer_type"),
            AtomicUnit(unit_id=3, source_span="the cat", function="target"),
        ]
        shards = [
            QuestionShard(
                shard_id=1,
                role="intent",
                text="Determine a property of a particular thing.",
                source_unit_ids=[1],
            ),
            QuestionShard(
                shard_id=2,
                role="answer_type",
                text="The requested property is color.",
                source_unit_ids=[2],
            ),
            QuestionShard(
                shard_id=3,
                role="target",
                text="The target thing is the cat.",
                source_unit_ids=[3],
            ),
        ]
        self.assertEqual(
            validate_shards("What color is the cat?", "Orange", units, shards),
            [],
        )
        self.assertEqual(
            shardability_label("What color is the cat?", 3),
            "structured_3_short",
        )

    def test_validate_plan_flags_answer_leakage_and_missing_coverage(self) -> None:
        units = [
            AtomicUnit(unit_id=1, source_span="What", function="intent"),
            AtomicUnit(unit_id=2, source_span="color", function="answer_type"),
            AtomicUnit(unit_id=3, source_span="the cat", function="target"),
        ]
        shards = [
            QuestionShard(
                shard_id=1,
                role="intent",
                text="Tell me whether the answer is orange.",
                source_unit_ids=[1],
            ),
            QuestionShard(
                shard_id=2,
                role="answer_type",
                text="The answer should be a color.",
                source_unit_ids=[2],
            ),
            QuestionShard(
                shard_id=3,
                role="target",
                text="Look at the animal.",
                source_unit_ids=[1],
            ),
        ]
        issues = validate_shards("What color is the cat?", "Orange", units, shards)
        self.assertIn("gold_answer_leakage", issues)
        self.assertIn("uncovered_atomic_units:3", issues)

    def test_all_category_prompt_bundles_exist(self) -> None:
        prompt_dir = REPO_ROOT / "configs" / "question_sharding"
        for category in CATEGORIES:
            bundle = read_prompt_bundle(prompt_dir, category)
            self.assertIn("Information preservation", bundle["base"])
            self.assertIn(f"CATEGORY: {category}", bundle["category"])
            self.assertIn("Minimal sufficiency", bundle["evidence_base"])
            self.assertIn(f"CATEGORY: {category}", bundle["evidence_category"])
            self.assertIn(
                "final independent verifier",
                bundle["integrated_verification"],
            )

    def test_validate_evidence_accepts_many_to_many_partition(self) -> None:
        shards = [
            QuestionShard(
                shard_id=1,
                role="intent",
                text="Determine a property of a particular object.",
                source_unit_ids=[1],
            ),
            QuestionShard(
                shard_id=2,
                role="answer_type",
                text="The requested property is color.",
                source_unit_ids=[1],
            ),
            QuestionShard(
                shard_id=3,
                role="target",
                text="The object belongs to the person beside the child.",
                source_unit_ids=[2, 3],
            ),
        ]
        plan = EvidencePlan(
            gold_layer_ids=[2, 4],
            layer_roles={"2": "answer_target", "4": "reference_landmark"},
            minimal_sufficient_layer_sets=[[2, 4]],
            question_shard_support={"1": [], "2": [2], "3": [2, 4]},
            evidence_dependencies=[
                EvidenceDependency(before=4, after=2, reason="reference_resolution")
            ],
            distractor_layer_ids=[0, 1, 3],
            evidence_notes="The landmark identifies the answer-bearing target.",
            confidence="high",
        )
        self.assertEqual(validate_evidence([0, 1, 2, 3, 4], shards, plan), [])

    def test_validate_evidence_flags_bad_partition_and_mapping(self) -> None:
        shards = [
            QuestionShard(
                shard_id=1,
                role="intent",
                text="Determine a quantity for particular things.",
                source_unit_ids=[1],
            ),
            QuestionShard(
                shard_id=2,
                role="answer_type",
                text="The requested quantity is a count.",
                source_unit_ids=[1],
            ),
            QuestionShard(
                shard_id=3,
                role="scope",
                text="The objects are bags.",
                source_unit_ids=[2],
            ),
        ]
        plan = EvidencePlan(
            gold_layer_ids=[1],
            layer_roles={"1": "count_member"},
            minimal_sufficient_layer_sets=[[1]],
            question_shard_support={"1": [], "2": [], "3": [2]},
            evidence_dependencies=[],
            distractor_layer_ids=[0],
            evidence_notes="Deliberately invalid fixture.",
            confidence="low",
        )
        issues = validate_evidence([0, 1, 2], shards, plan)
        self.assertIn("shard_support_contains_distractor", issues)
        self.assertIn("shard_support_does_not_cover_evidence_union", issues)
        self.assertIn("distractors_do_not_partition_candidates", issues)

    def test_temp_selection_is_balanced_and_deterministic(self) -> None:
        records = [
            {"question_id": f"{category}-{index}", "category": category}
            for category in CATEGORIES
            for index in range(3)
        ]
        first = select_temp_records(records, 6, 7)
        second = select_temp_records(records, 6, 7)
        self.assertEqual(
            [record["question_id"] for record in first],
            [record["question_id"] for record in second],
        )
        self.assertEqual({record["category"] for record in first}, set(CATEGORIES))

    def test_parse_categories_supports_repeated_and_comma_separated_values(
        self,
    ) -> None:
        self.assertEqual(
            parse_categories(["direct_ask,counting", "spatial"]),
            {"direct_ask", "counting", "spatial"},
        )
        with self.assertRaisesRegex(ValueError, "Unknown categories"):
            parse_categories(["not_a_category"])


if __name__ == "__main__":
    unittest.main()
