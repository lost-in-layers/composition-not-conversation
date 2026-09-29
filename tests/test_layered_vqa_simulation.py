"""End-to-end simulation checks for the layered_vqa task.

The OpenAI client is stubbed at the transport boundary rather than at
`model_openai.generate`, so these still exercise prompt population, local
image materialisation into data URLs, and the simulator's turn bookkeeping.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = REPO_ROOT / "vendor" / "conv-smi"
DATASET_DIR = REPO_ROOT / "data" / "derived" / "lic"

# The vendored code opens prompts by relative path, so imports and runs both
# need the vendor directory to be the working directory.
os.environ.setdefault("VLLM_BASE_URL", "http://localhost:1/v1")
os.environ.setdefault("VLLM_API_KEY", "stub")
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))


@contextmanager
def in_vendor_dir():
    """Run inside the vendor directory with a throwaway judge cache.

    The judge cache is persistent by design, so without isolation a verdict
    from one test leaks into the next and silently decides its outcome.
    """
    previous = Path.cwd()
    previous_cache = os.environ.get("LAYERED_VQA_JUDGE_CACHE")
    os.chdir(VENDOR_DIR)
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["LAYERED_VQA_JUDGE_CACHE"] = str(Path(tmp) / "judge_cache.jsonl")
        try:
            yield
        finally:
            os.chdir(previous)
            if previous_cache is None:
                os.environ.pop("LAYERED_VQA_JUDGE_CACHE", None)
            else:
                os.environ["LAYERED_VQA_JUDGE_CACHE"] = previous_cache


class FakeCompletion:
    def __init__(self, payload: str):
        self.payload = payload

    def to_dict(self):
        return {
            "choices": [{"message": {"content": self.payload}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }


class StubClient:
    """Answers each agent role by looking at the prompt it was handed."""

    def __init__(self, assistant_reply="The answer is blue.", judge_correct=True):
        self.assistant_reply = assistant_reply
        self.judge_correct = judge_correct
        self.calls = []
        self.image_urls = []

    def create(self, model, messages, **kwargs):
        self.calls.append(messages)
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image_url":
                        self.image_urls.append(part["image_url"]["url"])

        text = json.dumps(messages)
        if "grading an answer" in text:
            return FakeCompletion(
                json.dumps(
                    {
                        "reasoning": "stub",
                        "answer_span": self.assistant_reply if self.judge_correct else "",
                        "is_correct": self.judge_correct,
                    }
                )
            )
        return FakeCompletion(self.assistant_reply)


@contextmanager
def stubbed_model(assistant_reply="The answer is blue.", judge_correct=True):
    import model_openai

    stub = StubClient(assistant_reply, judge_correct)
    original = model_openai.model.client
    model_openai.model.client = type(
        "C", (), {"chat": type("Chat", (), {"completions": stub})()}
    )()
    try:
        yield stub
    finally:
        model_openai.model.client = original


def load_sample(dataset: str, task_id: str | None = None, category: str | None = None):
    samples = json.loads((DATASET_DIR / dataset).read_text())
    if task_id:
        return next(s for s in samples if s["task_id"] == task_id)
    if category:
        return next(s for s in samples if s["meta"]["category"] == category)
    return samples[0]


@unittest.skipUnless(
    (DATASET_DIR / "layered_vqa_text.json").exists(),
    "run src/vecgen/export_lic_dataset.py first",
)
class ShardedSimulationTests(unittest.TestCase):
    def test_scripted_agent_reveals_every_shard_in_dataset_order(self):
        sample = load_sample("layered_vqa_aligned_vq.json", category="multi_hop_reference")
        with in_vendor_dir(), stubbed_model("I am not sure yet.", judge_correct=False) as stub:
            os.environ["LAYERED_VQA_DATASET"] = str(
                DATASET_DIR / "layered_vqa_aligned_vq.json"
            )
            from simulator_sharded import ConversationSimulatorSharded

            sim = ConversationSimulatorSharded(
                sample,
                dataset_fn=str(DATASET_DIR / "layered_vqa_aligned_vq.json"),
                user_agent="scripted",
            )
            sim.run(verbose=False, save_log=False)

        revealed = [
            msg["content"]["shard_id"]
            for msg in sim.trace
            if msg["role"] == "log" and msg["content"]["type"] == "shard_revealed"
        ]
        self.assertEqual(revealed, [s["shard_id"] for s in sample["shards"]])
        self.assertEqual(sim.user_agent_kind, "scripted")

    def test_a_correct_answer_does_not_cut_the_conversation_short(self):
        """Grading is once, at the end — there is nothing to terminate early on."""
        sample = load_sample("layered_vqa_aligned_vq.json", category="multi_hop_reference")
        self.assertGreater(len(sample["shards"]), 1)

        with in_vendor_dir(), stubbed_model("The answer is blue.", judge_correct=True):
            os.environ["LAYERED_VQA_DATASET"] = str(
                DATASET_DIR / "layered_vqa_aligned_vq.json"
            )
            from simulator_sharded import ConversationSimulatorSharded

            sim = ConversationSimulatorSharded(
                sample,
                dataset_fn=str(DATASET_DIR / "layered_vqa_aligned_vq.json"),
                user_agent="scripted",
            )
            is_correct, score = sim.run(verbose=False, save_log=False)

        revealed = [
            m for m in sim.trace
            if m["role"] == "log" and m["content"]["type"] == "shard_revealed"
        ]
        log_types = [m["content"]["type"] for m in sim.trace if m["role"] == "log"]
        self.assertEqual(len(revealed), len(sample["shards"]))
        self.assertEqual(log_types.count("answer-evaluation"), 1)
        self.assertEqual(log_types.count("system-verification"), 0)
        self.assertTrue(is_correct)
        self.assertEqual(score, 1.0)

    def test_layer_images_reach_the_model_as_data_urls(self):
        sample = load_sample("layered_vqa_aligned_vq.json", category="multi_hop_reference")
        expected = sum(len(s.get("images", [])) for s in sample["shards"])
        self.assertGreater(expected, 0)

        with in_vendor_dir(), stubbed_model("I am not sure yet.", judge_correct=False) as stub:
            os.environ["LAYERED_VQA_DATASET"] = str(
                DATASET_DIR / "layered_vqa_aligned_vq.json"
            )
            from simulator_sharded import ConversationSimulatorSharded

            ConversationSimulatorSharded(
                sample,
                dataset_fn=str(DATASET_DIR / "layered_vqa_aligned_vq.json"),
                user_agent="scripted",
            ).run(verbose=False, save_log=False)

        self.assertTrue(stub.image_urls, "no images were sent to the model")
        for url in stub.image_urls:
            self.assertTrue(url.startswith("data:image/"), url[:40])
            base64.b64decode(url.split(",", 1)[1][:64] + "==", validate=False)

    def test_turn_text_names_the_layer_it_carries(self):
        sample = load_sample("layered_vqa_aligned_vq.json", category="multi_hop_reference")
        with in_vendor_dir(), stubbed_model("I am not sure yet.", judge_correct=False):
            os.environ["LAYERED_VQA_DATASET"] = str(
                DATASET_DIR / "layered_vqa_aligned_vq.json"
            )
            from simulator_sharded import ConversationSimulatorSharded
            from multimodal import content_to_text

            sim = ConversationSimulatorSharded(
                sample,
                dataset_fn=str(DATASET_DIR / "layered_vqa_aligned_vq.json"),
                user_agent="scripted",
            )
            sim.run(verbose=False, save_log=False)
            user_text = " ".join(
                content_to_text(m["content"]) for m in sim.trace if m["role"] == "user"
            )
        self.assertIn("(sharing layer", user_text)

    def test_traces_keep_readable_paths_not_base64(self):
        sample = load_sample("layered_vqa_aligned_vq.json", category="multi_hop_reference")
        with in_vendor_dir(), stubbed_model("I am not sure yet.", judge_correct=False):
            os.environ["LAYERED_VQA_DATASET"] = str(
                DATASET_DIR / "layered_vqa_aligned_vq.json"
            )
            from simulator_sharded import ConversationSimulatorSharded

            sim = ConversationSimulatorSharded(
                sample,
                dataset_fn=str(DATASET_DIR / "layered_vqa_aligned_vq.json"),
                user_agent="scripted",
            ).run(verbose=False, save_log=False)

        blob = json.dumps(
            [m for m in load_sample("layered_vqa_aligned_vq.json", category="multi_hop_reference")["shards"]]
        )
        self.assertNotIn("data:image", blob)


@unittest.skipUnless(
    (DATASET_DIR / "layered_vqa_text.json").exists(),
    "run src/vecgen/export_lic_dataset.py first",
)
class FullSimulationTests(unittest.TestCase):
    def test_full_run_scores_a_correct_answer(self):
        sample = load_sample("layered_vqa_text.json", category="direct_ask")
        gold = sample["answer"]
        with in_vendor_dir(), stubbed_model(f"The answer is {gold}"):
            os.environ["LAYERED_VQA_DATASET"] = str(
                DATASET_DIR / "layered_vqa_text.json"
            )
            from simulator_full import ConversationSimulatorFull

            is_correct, score = ConversationSimulatorFull(
                sample,
                assistant_model="local-vlm",
                dataset_fn=str(DATASET_DIR / "layered_vqa_text.json"),
            ).run(verbose=False, save_log=False)
        self.assertEqual(score, 1.0)
        self.assertTrue(is_correct)

    def test_full_run_scores_a_wrong_answer_zero(self):
        sample = load_sample("layered_vqa_text.json", category="direct_ask")
        wrong = "a wristwatch"
        with in_vendor_dir(), stubbed_model(f"The answer is {wrong}"):
            os.environ["LAYERED_VQA_DATASET"] = str(
                DATASET_DIR / "layered_vqa_text.json"
            )
            from simulator_full import ConversationSimulatorFull
            import model_openai

            # Route the judge to a definite "no" rather than the stub default.
            chat = model_openai.model.client.chat

            class Rejecting:
                def create(self, model, messages, **kwargs):
                    text = json.dumps(messages)
                    if "grading an answer" in text:
                        return FakeCompletion(
                            json.dumps({"reasoning": "different object", "is_correct": False})
                        )
                    return chat.completions.create(model, messages, **kwargs)

            model_openai.model.client = type(
                "C", (), {"chat": type("Chat", (), {"completions": Rejecting()})()}
            )()

            is_correct, score = ConversationSimulatorFull(
                sample,
                assistant_model="local-vlm",
                dataset_fn=str(DATASET_DIR / "layered_vqa_text.json"),
            ).run(verbose=False, save_log=False)
        self.assertEqual(score, 0.0)
        self.assertFalse(is_correct)


def grade(judge, transcript, reference, task_id):
    return judge.judge_conversation(
        transcript=transcript,
        reference=reference,
        question="What object?",
        answer_description="a short phrase",
        task_id=task_id,
    )


class JudgeTests(unittest.TestCase):
    def test_an_exact_match_reply_still_goes_to_the_llm_judge(self):
        """No deterministic shortcut: every non-empty transcript is the LLM's."""
        with in_vendor_dir(), stubbed_model("blue"):
            from tasks.layered_vqa.judge import AnswerJudge

            with tempfile.TemporaryDirectory() as tmp:
                judge = AnswerJudge(cache_path=Path(tmp) / "cache.jsonl")
                result = grade(judge, "[assistant] blue", "Blue.", "t1")
        self.assertEqual(result["grader"], "llm")
        self.assertEqual(result["score"], 1.0)

    def test_empty_conversation_fails_without_a_model_call(self):
        with in_vendor_dir():
            from tasks.layered_vqa.judge import AnswerJudge

            with tempfile.TemporaryDirectory() as tmp:
                judge = AnswerJudge(cache_path=Path(tmp) / "cache.jsonl")
                result = grade(judge, "", "Blue", "t1")
        self.assertEqual(result["grader"], "deterministic")
        self.assertEqual(result["score"], 0.0)

    def test_an_empty_final_reply_still_reaches_the_judge(self):
        """The answer may have been given on an earlier turn."""
        transcript = "[assistant] It is a mouse.\n\n[user] thanks\n\n[assistant] "
        with in_vendor_dir(), stubbed_model():
            from tasks.layered_vqa.judge import AnswerJudge

            with tempfile.TemporaryDirectory() as tmp:
                judge = AnswerJudge(cache_path=Path(tmp) / "cache.jsonl")
                result = grade(judge, transcript, "A computer mouse.", "t4")
        self.assertEqual(result["grader"], "llm")
        self.assertEqual(result["score"], 1.0)

    def test_llm_verdict_is_cached_and_reused(self):
        transcript = "[user] What object?\n\n[assistant] a mouse"
        with in_vendor_dir(), stubbed_model():
            from tasks.layered_vqa.judge import AnswerJudge

            with tempfile.TemporaryDirectory() as tmp:
                cache_path = Path(tmp) / "cache.jsonl"
                judge = AnswerJudge(cache_path=cache_path)
                first = grade(judge, transcript, "A computer mouse.", "t2")
                second = grade(judge, transcript, "A computer mouse.", "t2")

                reloaded = AnswerJudge(cache_path=cache_path)
                third = grade(reloaded, transcript, "A computer mouse.", "t2")

        self.assertEqual(first["grader"], "llm")
        self.assertEqual(second["grader"], "cache")
        self.assertEqual(third["grader"], "cache")
        self.assertTrue(all(r["score"] == 1.0 for r in (first, second, third)))

    def test_judge_outage_scores_zero_instead_of_crashing(self):
        with in_vendor_dir():
            from tasks.layered_vqa.judge import AnswerJudge
            import model_openai

            original = model_openai.model.client

            class Boom:
                def create(self, *a, **k):
                    raise RuntimeError("provider down")

            model_openai.model.client = type(
                "C", (), {"chat": type("Chat", (), {"completions": Boom()})()}
            )()
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    judge = AnswerJudge(cache_path=Path(tmp) / "cache.jsonl")
                    result = grade(judge, "[assistant] a mouse", "A cat.", "t3")
            finally:
                model_openai.model.client = original

        self.assertEqual(result["grader"], "error")
        self.assertEqual(result["score"], 0.0)


@contextmanager
def stubbed_replies(replies):
    """Stub the transport with a fixed sequence of raw completions."""
    import model_openai

    class SequenceClient:
        def __init__(self):
            self.replies = list(replies)
            self.calls = []

        def create(self, model, messages, **kwargs):
            self.calls.append(messages)
            return FakeCompletion(self.replies.pop(0))

    stub = SequenceClient()
    original = model_openai.model.client
    model_openai.model.client = type(
        "C", (), {"chat": type("Chat", (), {"completions": stub})()}
    )()
    try:
        yield stub
    finally:
        model_openai.model.client = original


class GenerateJsonTests(unittest.TestCase):
    """JSON mode does not guarantee parseable output, so `generate_json`
    must salvage or resample rather than crash the whole grading call —
    at temperature 1.0 roughly 1 in 400 judge replies came back with an
    unescaped quote inside a string, killing the run's log write."""

    def call(self, **kwargs):
        import model_openai

        return model_openai.generate_json(
            [{"role": "user", "content": "grade this"}], model="local-vlm", **kwargs
        )

    def test_a_malformed_reply_is_retried_with_a_fresh_sample(self):
        broken = '{"reasoning": "the assistant said "a mouse" early on", "is_correct": true}'
        good = '{"reasoning": "ok", "answer_span": "a mouse", "is_correct": true}'
        with in_vendor_dir(), stubbed_replies([broken, good]) as stub:
            result = self.call()
        self.assertEqual(len(stub.calls), 2)
        self.assertTrue(result["message"]["is_correct"])

    def test_a_fenced_reply_is_salvaged_without_a_second_call(self):
        fenced = 'Here is my verdict:\n```json\n{"is_correct": false}\n```'
        with in_vendor_dir(), stubbed_replies([fenced]) as stub:
            result = self.call()
        self.assertEqual(len(stub.calls), 1)
        self.assertEqual(result["message"], {"is_correct": False})

    def test_persistent_garbage_raises_after_the_retry_budget(self):
        with in_vendor_dir(), stubbed_replies(["not json at all"] * 2) as stub:
            with self.assertRaises(RuntimeError):
                self.call(max_retries=2)
        self.assertEqual(len(stub.calls), 2)


class ModelRoutingTests(unittest.TestCase):
    """models.yaml is the only thing deciding where a call goes."""

    def registry(self):
        from model_registry import ModelRegistry

        return ModelRegistry()

    def test_open_weights_go_to_vllm_and_closed_ones_to_openrouter(self):
        with in_vendor_dir():
            reg = self.registry()
            local = reg.resolve("local-vlm")
            hosted = reg.resolve("gpt-4o-mini")

        self.assertEqual(local.backend, "vllm")
        self.assertEqual(hosted.backend, "openrouter")
        # The alias is short; the wire id is the provider's.
        self.assertEqual(hosted.model_id, "openai/gpt-4o-mini")

    def test_an_unknown_model_is_an_error_not_a_guess(self):
        """Guessing costs money or fails confusingly; neither is acceptable."""
        with in_vendor_dir():
            reg = self.registry()
            with self.assertRaises(ValueError) as caught:
                reg.resolve("gpt-4o-miniature")
        self.assertIn("Unknown model", str(caught.exception))

    def test_inline_backend_id_skips_the_config(self):
        with in_vendor_dir():
            reg = self.registry()
            route = reg.resolve("openrouter:openai/gpt-4o")
            with self.assertRaises(ValueError):
                reg.resolve("nosuchbackend:some-model")

        self.assertEqual(route.backend, "openrouter")
        self.assertEqual(route.model_id, "openai/gpt-4o")

    def test_the_wire_call_uses_the_resolved_id(self):
        sent = {}

        class Recorder:
            def create(self, model, messages, **kwargs):
                sent["model"] = model
                return FakeCompletion("ok")

        with in_vendor_dir():
            import model_openai

            model = model_openai.OpenAI_Model()
            model.client = type("C", (), {"chat": type("Ch", (), {"completions": Recorder()})()})()
            model.generate([{"role": "user", "content": "hi"}], model="qwen2.5-vl-7b")

        self.assertEqual(sent["model"], "Qwen/Qwen2.5-VL-7B-Instruct")


class TranscriptTests(unittest.TestCase):
    def test_image_parts_render_without_their_paths(self):
        with in_vendor_dir():
            from tasks.layered_vqa.task_layered_vqa import render_transcript

            trace = [
                {"role": "system", "content": "system prompt"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "here's another part"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "../lic_renders/scene/layer_2.png"},
                        },
                    ],
                },
                {"role": "assistant", "content": "a mouse"},
                {"role": "log", "content": {"type": "shard_revealed", "shard_id": 2}},
            ]
            transcript = render_transcript(trace)

        self.assertNotIn("layer_2.png", transcript)
        self.assertIn("[image]", transcript)
        self.assertNotIn("system prompt", transcript)   # skip_system
        self.assertNotIn("shard_revealed", transcript)  # log entries are not turns
        self.assertIn("[assistant] a mouse", transcript)


class LayerSelectionParsingTests(unittest.TestCase):
    """Reading the `Answer:` / `Layers:` / `Confidence:` reply of condition 2."""

    def parse(self, reply):
        with in_vendor_dir():
            from tasks.layered_vqa.layer_selection import parse_layer_selection

            return parse_layer_selection(reply)

    def test_a_well_formed_reply_is_read_field_by_field(self):
        parsed = self.parse(
            "Answer: a computer mouse\n"
            "Layers: 3, 5\n"
            "Confidence: 0.8\n"
            "Why: layer 3 shows the object, layer 5 disambiguates it.\n"
        )
        self.assertEqual(parsed["answer"], "a computer mouse")
        self.assertEqual(parsed["layer_ids"], [3, 5])
        self.assertEqual(parsed["confidence"], 0.8)
        self.assertTrue(parsed["format_ok"])
        self.assertIn("layer 3 shows", parsed["explanation"])

    def test_prose_and_markdown_around_the_ids_still_parse(self):
        parsed = self.parse("**Answer:** blue\n- Layers: layer 3 and layer 5\nConfidence: 80%")
        self.assertEqual(parsed["answer"], "blue")
        self.assertEqual(parsed["layer_ids"], [3, 5])
        self.assertEqual(parsed["confidence"], 0.8)

    def test_a_restated_block_is_read_from_its_last_occurrence(self):
        parsed = self.parse("Answer: blue\nLayers: 1\n\nOn reflection:\nAnswer: green\nLayers: 2")
        self.assertEqual(parsed["answer"], "green")
        self.assertEqual(parsed["layer_ids"], [2])

    def test_repeated_ids_collapse_and_keep_their_order(self):
        self.assertEqual(self.parse("Layers: 5, 3, 5")["layer_ids"], [5, 3])

    def test_a_label_in_parentheses_does_not_add_an_id(self):
        parsed = self.parse("Layers: layer 0 (the background layer), layer 2")
        self.assertEqual(parsed["layer_ids"], [0, 2])

    def test_a_separator_is_never_read_as_a_negative_id(self):
        # "1-3" is a sloppy range, not layer -3; ids are non-negative.
        self.assertEqual(self.parse("Layers: 1-3")["layer_ids"], [1, 3])

    def test_an_explicit_none_is_an_empty_prediction(self):
        parsed = self.parse("Answer: blue\nLayers: none\n")
        self.assertEqual(parsed["layer_ids"], [])
        self.assertTrue(parsed["format_ok"])

    def test_a_missing_layers_line_is_a_format_failure_not_an_empty_set(self):
        # An empty set claims "no layer supported this"; a missing line claims
        # nothing. Scoring them alike would report every non-compliant model as
        # a confidently ungrounded one.
        parsed = self.parse("The bulb is blue.")
        self.assertIsNone(parsed["layer_ids"])
        self.assertFalse(parsed["format_ok"])
        self.assertIsNone(parsed["answer"])

    def test_an_unreadable_layers_line_is_a_format_failure(self):
        parsed = self.parse("Answer: blue\nLayers: the background one\n")
        self.assertIsNone(parsed["layer_ids"])
        self.assertFalse(parsed["format_ok"])
        self.assertIn("could not read", parsed["parse_error"])

    def test_an_out_of_range_confidence_is_dropped(self):
        self.assertIsNone(self.parse("Layers: 1\nConfidence: high")["confidence"])
        self.assertIsNone(self.parse("Layers: 1\nConfidence: -2")["confidence"])


class LayerMetricTests(unittest.TestCase):
    def metrics(self, predicted, **overrides):
        with in_vendor_dir():
            from tasks.layered_vqa.layer_selection import layer_metrics

            kwargs = dict(
                gold=[2, 3],
                minimal_sets=[[2, 3]],
                distractors=[0, 1],
                candidates=[0, 1, 2, 3],
            )
            kwargs.update(overrides)
            return layer_metrics(predicted, **kwargs)

    def test_an_exact_evidence_set_scores_everything(self):
        metrics = self.metrics([3, 2])
        self.assertEqual((metrics["precision"], metrics["recall"], metrics["f1"]), (1.0, 1.0, 1.0))
        self.assertTrue(metrics["exact_match"])
        self.assertTrue(metrics["minimal_set_success"])
        self.assertTrue(metrics["complete"])
        self.assertEqual(metrics["excess"], 0)
        self.assertFalse(metrics["distractor_captured"])

    def test_a_distractor_costs_precision_but_not_completeness(self):
        metrics = self.metrics([2, 3, 1])
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertTrue(metrics["complete"])
        self.assertFalse(metrics["exact_match"])
        self.assertEqual(metrics["distractors_selected"], 1)
        self.assertEqual(metrics["excess"], 1)

    def test_a_missing_gold_layer_costs_recall_and_completeness(self):
        metrics = self.metrics([2])
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertFalse(metrics["complete"])

    def test_an_invented_id_is_reported_separately_from_a_wrong_one(self):
        metrics = self.metrics([2, 3, 9])
        self.assertEqual(metrics["invalid_ids"], [9])
        self.assertEqual(metrics["distractors_selected"], 0)
        self.assertEqual(metrics["excess"], 1)

    def test_a_second_accepted_minimal_set_also_counts_as_success(self):
        metrics = self.metrics([3], gold=[2], minimal_sets=[[2], [3]])
        self.assertTrue(metrics["minimal_set_success"])
        self.assertFalse(metrics["exact_match"])  # graded against the gold set

    def test_an_empty_prediction_scores_zero_rather_than_dividing_by_zero(self):
        metrics = self.metrics([])
        self.assertEqual((metrics["precision"], metrics["recall"], metrics["f1"]), (0.0, 0.0, 0.0))
        self.assertEqual(metrics["excess_rate"], 0.0)


@unittest.skipUnless(
    (DATASET_DIR / "layered_vqa_grounding.json").exists(),
    "run src/vecgen/export_lic_dataset.py first",
)
class GroundingConditionTests(unittest.TestCase):
    """Experiment A condition 2, end to end through the full simulator."""

    def run_full(self, dataset, reply, task_id=None, category=None):
        sample = load_sample(dataset, task_id=task_id, category=category)
        with in_vendor_dir(), stubbed_model(reply) as stub:
            os.environ["LAYERED_VQA_DATASET"] = str(DATASET_DIR / dataset)
            from simulator_full import ConversationSimulatorFull

            simulator = ConversationSimulatorFull(
                sample,
                assistant_model="local-vlm",
                dataset_fn=str(DATASET_DIR / dataset),
            )
            trace = []
            original = simulator.task.evaluate_conversation

            def capture(conversation, sample_):
                result = original(conversation, sample_)
                trace.append((conversation, result))
                return result

            simulator.task.evaluate_conversation = capture
            is_correct, score = simulator.run(verbose=False, save_log=False)
        return sample, stub, trace[-1], (is_correct, score)

    def test_the_prompt_asks_for_layers_without_numbering_the_manifest(self):
        sample, stub, (conversation, _), _ = self.run_full(
            "layered_vqa_grounding.json", "Answer: blue\nLayers: 1\nConfidence: 0.5\nWhy: it shows the bulb."
        )
        prompt = json.dumps(stub.calls[0])
        self.assertIn("Layers:", prompt)
        # Positions would collide with the ids the model is asked to cite.
        self.assertIn("- layer", prompt)
        self.assertNotIn("1. the full image", prompt)
        # Full RGB plus every retained layer, in one turn.
        self.assertEqual(len(stub.image_urls), len(sample["images"]))
        self.assertTrue(all(url.startswith("data:image/") for url in stub.image_urls))

    def test_a_correct_answer_with_the_gold_set_is_jointly_correct(self):
        sample = load_sample("layered_vqa_grounding.json")
        gold = sample["meta"]["gold_layer_ids"]
        reply = (
            f"Answer: {sample['answer']}\n"
            f"Layers: {', '.join(str(i) for i in gold)}\n"
            "Confidence: 0.9\nWhy: those layers show it.\n"
        )
        _, _, (_, evaluation), (is_correct, _) = self.run_full(
            "layered_vqa_grounding.json", reply, task_id=sample["task_id"]
        )
        self.assertTrue(is_correct)
        self.assertEqual(evaluation["layer_selection"]["layer_ids"], gold)
        self.assertTrue(evaluation["layer_metrics"]["exact_match"])
        self.assertTrue(evaluation["joint_correct"])
        self.assertTrue(evaluation["joint_complete"])
        self.assertEqual(evaluation["grader"], "llm")

    def test_a_correct_answer_on_the_wrong_layers_is_not_jointly_correct(self):
        sample = load_sample("layered_vqa_grounding.json")
        wrong = [i for i in sample["meta"]["candidate_layer_ids"] if i not in sample["meta"]["gold_layer_ids"]]
        reply = f"Answer: {sample['answer']}\nLayers: {wrong[0]}\nConfidence: 0.4\nWhy: guessing.\n"
        _, _, (_, evaluation), (is_correct, _) = self.run_full(
            "layered_vqa_grounding.json", reply, task_id=sample["task_id"]
        )
        self.assertTrue(is_correct)
        self.assertFalse(evaluation["layer_metrics"]["exact_match"])
        self.assertFalse(evaluation["joint_correct"])

    def test_an_unreadable_reply_is_a_format_failure_not_a_zero_scored_set(self):
        sample = load_sample("layered_vqa_grounding.json")
        _, _, (_, evaluation), _ = self.run_full(
            "layered_vqa_grounding.json",
            f"It is {sample['answer']}.",
            task_id=sample["task_id"],
        )
        self.assertFalse(evaluation["layer_selection"]["format_ok"])
        self.assertIsNone(evaluation["layer_metrics"])
        self.assertFalse(evaluation["joint_correct"])

    def test_the_verdict_reaches_the_log_through_the_trace(self):
        sample = load_sample("layered_vqa_grounding.json")
        reply = f"Answer: {sample['answer']}\nLayers: {sample['meta']['gold_layer_ids'][0]}\nConfidence: 0.7\nWhy: shown.\n"
        _, _, (conversation, _), _ = self.run_full(
            "layered_vqa_grounding.json", reply, task_id=sample["task_id"]
        )
        entry = next(
            m for m in conversation
            if m["role"] == "log" and m["content"]["type"] == "answer-evaluation"
        )
        self.assertIn("layer_metrics", entry["content"]["evaluation_return"])


@unittest.skipUnless(
    (DATASET_DIR / "layered_vqa_oracle_gold.json").exists(),
    "run src/vecgen/export_lic_dataset.py first",
)
class OracleConditionTests(unittest.TestCase):
    """Experiment A condition 3 stays an answer-only run."""

    def test_only_the_gold_layers_are_sent_and_no_layers_are_requested(self):
        sample = load_sample("layered_vqa_oracle_gold.json")
        with in_vendor_dir(), stubbed_model(f"It is {sample['answer']}.") as stub:
            os.environ["LAYERED_VQA_DATASET"] = str(DATASET_DIR / "layered_vqa_oracle_gold.json")
            from simulator_full import ConversationSimulatorFull

            is_correct, _ = ConversationSimulatorFull(
                sample,
                assistant_model="local-vlm",
                dataset_fn=str(DATASET_DIR / "layered_vqa_oracle_gold.json"),
            ).run(verbose=False, save_log=False)

        self.assertTrue(is_correct)
        self.assertEqual(len(stub.image_urls), len(sample["meta"]["oracle_layer_ids"]))
        prompt = json.dumps(stub.calls[0])
        self.assertNotIn("Layers:", prompt)
        # The answer-only prompt, not the grounding one (5905aaf standardized
        # the wording; this asserts the prompt choice, not the phrasing).
        self.assertIn("Return only the answer, normally as a short phrase", prompt)


@unittest.skipUnless(
    (DATASET_DIR / "layered_vqa_visual_canonical.json").exists(),
    "run src/vecgen/export_lic_dataset.py first",
)
class SnowballSimulationTests(unittest.TestCase):
    def test_snowball_reattaches_every_revealed_layer_each_turn(self):
        sample = load_sample("layered_vqa_visual_canonical.json")
        with in_vendor_dir(), stubbed_model("I am not sure yet.", judge_correct=False):
            os.environ["LAYERED_VQA_DATASET"] = str(
                DATASET_DIR / "layered_vqa_visual_canonical.json"
            )
            from simulator_snowball import ConversationSimulatorSnowball

            sim = ConversationSimulatorSnowball(
                "layered_vqa", sample, user_agent="scripted"
            )
            sim.run(verbose=False, save_log=False)

        image_counts = []
        for msg in sim.trace:
            if msg["role"] != "user":
                continue
            content = msg["content"]
            image_counts.append(
                sum(1 for p in content if isinstance(p, dict) and p.get("type") == "image_url")
                if isinstance(content, list)
                else 0
            )
        # Shard 1 is the bare question; every later turn re-attaches all
        # layers revealed so far plus the new one: 0, 1, 2, ..., n-1.
        self.assertEqual(image_counts, list(range(len(sample["shards"]))))


@unittest.skipUnless(
    (DATASET_DIR / "layered_vqa_visual_canonical.json").exists(),
    "run src/vecgen/export_lic_dataset.py first",
)
class RecapSimulationTests(unittest.TestCase):
    def test_recap_v_resends_the_full_rgb_not_the_layers(self):
        sample = load_sample("layered_vqa_visual_canonical.json")
        layer_count = sum(len(s.get("images", [])) for s in sample["shards"])
        self.assertGreater(layer_count, 1)

        sharded_log = {
            "conv_id": "conv-under-test",
            "conv_type": "sharded-scripted",
            "task_id": sample["task_id"],
            "assistant_model": "local-vlm",
            "user_model": "local-vlm",
            "is_correct": False,
            "trace": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": sample["shards"][0]["shard"]},
                {"role": "assistant", "content": "I need more information."},
            ],
        }

        with in_vendor_dir(), stubbed_model("The answer is unclear.", judge_correct=False) as stub:
            os.environ["LAYERED_VQA_DATASET"] = str(
                DATASET_DIR / "layered_vqa_visual_canonical.json"
            )
            from simulator_recap import RecapSimulator

            RecapSimulator("layered_vqa").run_recap_sample(
                sharded_log, "recap-concat", save_log=False
            )

        recap_turn = stub.calls[0][-1]
        self.assertEqual(recap_turn["role"], "user")
        parts = recap_turn["content"]
        self.assertIsInstance(parts, list)
        images = [p for p in parts if p.get("type") == "image_url"]
        text = "\n".join(p["text"] for p in parts if p.get("type") == "text")
        # One image — the full RGB — even though the variant shards carry many.
        self.assertEqual(len(images), 1)
        self.assertIn("the full image", text)
        # The visual variant recaps the actual question, not filler captions.
        self.assertIn(sample["question"], text)
        self.assertNotIn("here's another part of the image", text)

    def test_the_judge_sees_the_recap_exchange_and_not_the_replayed_turns(self):
        """RECAP measures recovery, so the replayed conversation is off-limits.

        Sharded runs no longer stop when they answer correctly, so a judge
        shown the whole trace would credit the recap for the original
        conversation's answer.
        """
        sample = load_sample("layered_vqa_visual_canonical.json")
        sharded_log = {
            "conv_id": "conv-under-test",
            "conv_type": "sharded-scripted",
            "task_id": sample["task_id"],
            "assistant_model": "local-vlm",
            "user_model": "local-vlm",
            "is_correct": False,
            "trace": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": sample["shards"][0]["shard"]},
                {"role": "assistant", "content": f"It is {sample['answer']}."},
            ],
        }

        with in_vendor_dir(), stubbed_model("The answer is unclear.", judge_correct=False) as stub:
            os.environ["LAYERED_VQA_DATASET"] = str(
                DATASET_DIR / "layered_vqa_visual_canonical.json"
            )
            from simulator_recap import RecapSimulator

            RecapSimulator("layered_vqa").run_recap_sample(
                sharded_log, "recap-concat", save_log=False
            )

        judge_prompts = [
            json.dumps(messages) for messages in stub.calls
            if "grading an answer" in json.dumps(messages)
        ]
        self.assertEqual(len(judge_prompts), 1)
        self.assertIn("The answer is unclear.", judge_prompts[0])
        self.assertNotIn("It is ", judge_prompts[0])


@unittest.skipUnless(
    (DATASET_DIR / "layered_vqa_text.json").exists(),
    "run src/vecgen/export_lic_dataset.py first",
)
class ShuffleConcatTests(unittest.TestCase):
    def test_intent_stays_first_and_the_sample_is_not_mutated(self):
        sample = load_sample("layered_vqa_text.json")
        original_order = [s["shard_id"] for s in sample["shards"]]
        with in_vendor_dir(), stubbed_model("The answer is blue.", judge_correct=False) as stub:
            os.environ["LAYERED_VQA_DATASET"] = str(DATASET_DIR / "layered_vqa_text.json")
            from simulator_full import ConversationSimulatorFull

            ConversationSimulatorFull(
                sample,
                assistant_model="local-vlm",
                run_shuffle_concat=True,
                dataset_fn=str(DATASET_DIR / "layered_vqa_text.json"),
            ).run(verbose=False, save_log=False)

        # The queued sample dict is shared across conditions; shuffling it in
        # place would corrupt concurrently queued runs.
        self.assertEqual([s["shard_id"] for s in sample["shards"]], original_order)

        user_turn = stub.calls[0][-1]["content"]
        text = (
            user_turn
            if isinstance(user_turn, str)
            else "\n".join(p["text"] for p in user_turn if p.get("type") == "text")
        )
        bullets = [line for line in text.splitlines() if line.startswith("- ")]
        self.assertEqual(len(bullets), len(sample["shards"]))
        self.assertEqual(bullets[0], f"- {sample['shards'][0]['shard']}")


class ScriptedAgentTests(unittest.TestCase):
    def test_agent_returns_shards_in_order_then_stops_adding(self):
        with in_vendor_dir():
            from user_agent_scripted import ScriptedUserAgent

            class FakeTask:
                def get_task_name(self):
                    return "layered_vqa"

            agent = ScriptedUserAgent(FakeTask())
            sample = {"shards": [{"shard_id": 1, "shard": "a"}, {"shard_id": 2, "shard": "b"}]}

            trace = []
            text, shard_id = agent.generate_response(trace, sample)
            self.assertEqual((text, shard_id), ("a", 1))

            trace.append({"role": "log", "content": {"type": "shard_revealed", "shard_id": 1}})
            text, shard_id = agent.generate_response(trace, sample)
            self.assertEqual((text, shard_id), ("b", 2))

            trace.append({"role": "log", "content": {"type": "shard_revealed", "shard_id": 2}})
            _, shard_id = agent.generate_response(trace, sample)
            self.assertEqual(shard_id, -1)

    def test_image_datasets_need_an_explicit_agent_choice(self):
        """Reveal order is part of the design there, so no silent default —
        but an explicit llm choice is a legal (improvised-order) arm."""
        with in_vendor_dir():
            from user_agent_scripted import resolve_user_agent

            visual = [{"task_id": "t", "shards": [{"shard_id": 1, "shard": "q"}, {"shard_id": 2, "shard": "here's another part", "images": [{"path": "layer_2.png"}]}]}]
            text_only = [{"task_id": "t", "shards": [{"shard_id": 1, "shard": "q"}, {"shard_id": 2, "shard": "the colour"}]}]

            with self.assertRaises(ValueError):
                resolve_user_agent(None, visual)
            self.assertEqual(resolve_user_agent("llm", visual), "llm")
            self.assertEqual(resolve_user_agent("scripted", visual), "scripted")
            self.assertEqual(resolve_user_agent("scripted-shuffle", visual), "scripted-shuffle")
            self.assertEqual(resolve_user_agent(None, text_only), "llm")

    def test_shuffle_agent_keeps_intent_first_and_reveals_everything(self):
        with in_vendor_dir():
            from user_agent_scripted import build_user_agent

            class FakeTask:
                def get_task_name(self):
                    return "layered_vqa"

            agent = build_user_agent("scripted-shuffle", FakeTask(), model=None)
            self.assertTrue(agent.shuffle)
            sample = {
                "shards": [
                    {"shard_id": i, "shard": f"shard {i}"} for i in range(1, 6)
                ]
            }

            trace = []
            revealed = []
            while True:
                _, shard_id = agent.generate_response(trace, sample)
                if shard_id == -1:
                    break
                revealed.append(shard_id)
                trace.append(
                    {"role": "log", "content": {"type": "shard_revealed", "shard_id": shard_id}}
                )

        self.assertEqual(revealed[0], 1)  # intent always opens
        self.assertEqual(sorted(revealed), [1, 2, 3, 4, 5])


class LlmUserAgentTests(unittest.TestCase):
    """The improvising agent's bookkeeping and its view of the conversation."""

    SAMPLE = {
        "shards": [
            {"shard_id": 1, "shard": "intent"},
            {"shard_id": 2, "shard": "detail A"},
            {"shard_id": 3, "shard": "detail B", "images": [{"path": "../lic_renders/scene/layer_2.png", "label": "layer 2"}]},
        ]
    }

    def run_second_turn(self, reply, conversation):
        """Drive one post-intent turn with a stubbed user-model response."""
        with in_vendor_dir():
            import user_agent as user_agent_module

            captured = {}

            def fake_generate_json(messages, **kwargs):
                captured["prompt"] = messages[0]["content"]
                return {"message": reply}

            class FakeTask:
                def get_task_name(self):
                    return "layered_vqa"

            original = user_agent_module.generate_json
            user_agent_module.generate_json = fake_generate_json
            try:
                agent = user_agent_module.UserAgent(FakeTask(), model="stub")
                text, shard_id = agent.generate_response(conversation, self.SAMPLE)
            finally:
                user_agent_module.generate_json = original
        return text, shard_id, captured["prompt"]

    def hint_conversation(self):
        """A snowball-style trace: reveals are logged as `hint_revealed`."""
        return [
            {"role": "user", "content": "intent"},
            {"role": "log", "content": {"type": "hint_revealed", "shard_id": 1}},
            {"role": "assistant", "content": "which object?"},
            {"role": "user", "content": "detail A"},
            {"role": "log", "content": {"type": "hint_revealed", "shard_id": 2}},
            {"role": "assistant", "content": "hmm"},
        ]

    def test_snowball_hints_count_as_revealed(self):
        _, shard_id, prompt = self.run_second_turn(
            {"response": "ok", "shard_id": 3}, self.hint_conversation()
        )
        self.assertEqual(shard_id, 3)
        revealed_section, not_revealed_section = prompt.split("not been revealed yet:")
        self.assertIn("detail A", revealed_section)
        self.assertIn("detail B", not_revealed_section)
        self.assertNotIn("detail A", not_revealed_section)

    def test_model_supplied_shard_ids_are_sanitized(self):
        """The loop guards trust these ids, so anything off maps to -1."""
        conversation = self.hint_conversation()

        _, shard_id, _ = self.run_second_turn({"response": "x", "shard_id": "3"}, conversation)
        self.assertEqual(shard_id, 3)  # string id from JSON is coerced

        _, shard_id, _ = self.run_second_turn({"response": "x", "shard_id": 99}, conversation)
        self.assertEqual(shard_id, -1)  # not a shard of this sample

        _, shard_id, _ = self.run_second_turn({"response": "x", "shard_id": 2}, conversation)
        self.assertEqual(shard_id, -1)  # already revealed

        _, shard_id, _ = self.run_second_turn({"response": "x", "shard_id": "??"}, conversation)
        self.assertEqual(shard_id, -1)  # not an integer at all

    def test_image_shards_advertise_labels_never_paths(self):
        """The agent needs to know a shard brings an image to choose it
        deliberately, but only by its neutral label — filenames name the answer."""
        _, _, prompt = self.run_second_turn(
            {"response": "ok", "shard_id": 3}, self.hint_conversation()
        )
        self.assertIn("attaches", prompt)
        self.assertIn("layer 2", prompt)
        self.assertNotIn("layer_2.png", prompt)

    def test_conversation_view_hides_image_paths(self):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "here's another part"},
                    {"type": "image_url", "image_url": {"url": "../lic_renders/scene/layer_2.png", "detail": "high"}},
                ],
            },
            {"role": "log", "content": {"type": "shard_revealed", "shard_id": 1}},
            {"role": "assistant", "content": "which layer?"},
        ]
        _, _, prompt = self.run_second_turn({"response": "ok", "shard_id": 2}, conversation)
        self.assertNotIn("layer_2.png", prompt)
        self.assertIn("[image]", prompt)


class GradeConversationTests(unittest.TestCase):
    def test_a_judge_outage_aborts_the_run_instead_of_logging_zero(self):
        """A logged run is a completed run to the resume accounting, so a
        judge failure must raise before the caller reaches its log write."""
        with in_vendor_dir():
            from utils import JudgeUnavailableError, grade_conversation

            class BrokenJudgeTask:
                def evaluate_conversation(self, trace, sample):
                    return {
                        "score": 0.0,
                        "is_correct": False,
                        "grader": "error",
                        "answer_span": "",
                        "error": "RuntimeError: provider down",
                    }

            trace = [{"role": "assistant", "content": "a mouse"}]
            with self.assertRaises(JudgeUnavailableError):
                grade_conversation(BrokenJudgeTask(), trace, sample={})
        self.assertEqual(len(trace), 1)  # no verdict entry: nothing to log


if __name__ == "__main__":
    unittest.main()
