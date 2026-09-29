import json
import os
from typing import Any, Dict, List

from multimodal import all_sample_images
from task_base import Task
from utils import extract_conversation

from .judge import AnswerJudge
from .layer_selection import grounding_report

DEFAULT_DATASET = "../../data/derived/lic/layered_vqa_text.json"


def render_transcript(trace: List[Dict[str, Any]]) -> str:
    """Render a conversation as the text the judge reads.

    Image parts come out as a bare `[image]`, never `[image: <path>]` — image
    paths must not reach an agent prompt, and the judge is text-only, so the
    path would be noise even if it were safe. The user agent renders its
    conversation view through the same `hide_image_paths` flag; keeping both
    on one renderer is what stops the two from drifting apart again.
    """
    return extract_conversation(trace, to_str=True, skip_system=True, hide_image_paths=True)


def last_assistant_reply(trace: List[Dict[str, Any]]) -> str:
    replies = [m["content"] for m in trace if m["role"] == "assistant"]
    return replies[-1] if replies and isinstance(replies[-1], str) else ""


class TaskLayeredVQA(Task):
    """Layer-grounded visual question answering over decomposed scenes.

    Samples come from `src/vecgen/export_lic_dataset.py`. Which exported
    variant is loaded is set by the `LAYERED_VQA_DATASET` environment variable
    so one task class can serve every condition; `simulator_recap` reloads
    samples through `get_sample`, so it has to point at the same file the run
    was launched with.
    """

    def __init__(self):
        with open("prompts/layered_vqa/layered_vqa_system_prompt.txt", "r", encoding="utf-8") as f:
            self.system_prompt = f.read()
        with open("prompts/layered_vqa/layered_vqa_full_prompt.txt", "r", encoding="utf-8") as f:
            self.fully_specified_prompt = f.read()
        # Experiment A condition 2 asks for the answer *and* the supporting
        # layer ids; `meta.response_format` on the sample selects it.
        with open("prompts/layered_vqa/layered_vqa_grounding_prompt.txt", "r", encoding="utf-8") as f:
            self.grounding_prompt = f.read()

        self.judge = AnswerJudge()
        self._samples: List[Dict[str, Any]] | None = None

    def get_task_name(self) -> str:
        return "layered_vqa"

    def get_dataset_file(self) -> str:
        return os.environ.get("LAYERED_VQA_DATASET", DEFAULT_DATASET)

    def get_samples(self, filter: str | None = None) -> List[Dict[str, Any]]:
        if self._samples is None:
            with open(self.get_dataset_file(), "r", encoding="utf-8") as f:
                self._samples = json.load(f)
        if filter in (None, "full", "all"):
            return self._samples
        return [s for s in self._samples if s["meta"].get("category") == filter]

    def get_answer_description(self) -> str:
        return (
            "The answer should be a short phrase naming an object, an attribute, "
            "a count, or a spatial relation (for example 'blue', 'a computer mouse', "
            "'two', 'to the left of the chair')."
        )

    def generate_system_prompt(self, sample: Dict[str, Any]) -> str:
        return self.system_prompt

    # -- prompt construction -------------------------------------------------

    @staticmethod
    def asks_for_layers(sample: Dict[str, Any]) -> bool:
        return sample.get("meta", {}).get("response_format") == "answer_and_layers"

    def image_manifest(self, sample: Dict[str, Any]) -> str:
        """Name the images in the exact order the simulator will attach them.

        Content parts carry no filenames, so a turn holding several layers is
        otherwise unidentifiable. `all_sample_images` defines the order: the
        sample-level images first, then each shard's images in shard order.
        """
        return self._manifest_for(
            all_sample_images(sample), numbered=not self.asks_for_layers(sample)
        )

    @staticmethod
    def _manifest_for(images: List[Any], numbered: bool = True) -> str:
        """Render the manifest, optionally without position numbers.

        Numbering is dropped whenever the model is asked to cite layer ids: a
        list reading `2. layer 0 (the background layer)` offers two different
        numbers for the same image, and a model that answers with the position
        instead of the id produces a wrong evidence set on every sample.
        """
        labels = [
            image.get("label") if isinstance(image, dict) else None
            for image in images
        ]
        if not labels or not any(labels):
            return ""

        lines = ["The images below are, in order:"]
        for position, label in enumerate(labels, start=1):
            marker = f"{position}." if numbered else "-"
            lines.append(f"  {marker} {label or 'an image'}")
        return "\n".join(lines) + "\n"

    def _question_template(self, sample: Dict[str, Any]) -> str:
        return (
            self.grounding_prompt
            if self.asks_for_layers(sample)
            else self.fully_specified_prompt
        )

    def populate_fully_specific_prompt(self, sample: Dict[str, Any]) -> str:
        return self._question_template(sample).replace(
            "[[IMAGE_MANIFEST]]", self.image_manifest(sample)
        ).replace("[[QUESTION]]", sample["question"])

    def populate_concat_prompt(self, sample: Dict[str, Any]) -> str:
        query = "\n".join(f"- {shard['shard']}" for shard in sample["shards"])
        return self._question_template(sample).replace(
            "[[IMAGE_MANIFEST]]", self.image_manifest(sample)
        ).replace("[[QUESTION]]", query)

    def populate_recap_prompt(self, sample: Dict[str, Any]) -> tuple[str, List[Any]]:
        """The recap turn re-states the task and re-sends the full RGB image.

        RECAP_Q / RECAP_V are defined as showing the full RGB again — never
        the per-shard layer renders, which is what `all_sample_images` would
        re-attach on visual variants. The exporter writes that recap set as
        `recap_images`. Text-carrying variants restate their shards; visual
        variants restate the full question, because their shards are filler
        captions ("here's another part of the image") that recap nothing.
        """
        images = list(sample.get("recap_images") or all_sample_images(sample))
        if sample.get("meta", {}).get("variant") == "visual":
            query = sample["question"]
        else:
            query = "\n".join(f"- {shard['shard']}" for shard in sample["shards"])
        text = self.fully_specified_prompt.replace(
            "[[IMAGE_MANIFEST]]", self._manifest_for(images)
        ).replace("[[QUESTION]]", query)
        return text, images

    # -- evaluation ----------------------------------------------------------

    def evaluate_conversation(
        self, trace: List[Dict[str, Any]], sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Grade a finished conversation from its transcript, in one call.

        On layer-selection samples the grounding is scored here too, and the
        result rides back inside this dict: `grade_conversation` writes the
        whole return value into the trace's `answer-evaluation` entry, so no
        simulator has to know about Experiment A. The layer metrics are
        deterministic and recomputed every run — the judge cache holds only the
        answer verdict.
        """
        reply = last_assistant_reply(trace)
        report = grounding_report(sample, reply) if self.asks_for_layers(sample) else None

        result = self.judge.judge_conversation(
            transcript=render_transcript(trace),
            reference=sample["answer"],
            question=sample["question"],
            answer_description=self.get_answer_description(),
            task_id=sample["task_id"],
        )

        if report is not None:
            metrics = report["layer_metrics"]
            is_correct = bool(result.get("is_correct"))
            result.update(report)
            # Joint accuracy: a correct answer only counts when the evidence
            # behind it is right too. A reply whose layers could not be read
            # scores neither, and is counted as a format failure instead.
            result["joint_correct"] = is_correct and bool(metrics and metrics["exact_match"])
            result["joint_complete"] = is_correct and bool(metrics and metrics["complete"])

        return result

    def process_original_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        meta = sample.get("meta", {})
        return {
            "task_id": sample["task_id"],
            "question": sample["question"],
            "answer": sample["answer"],
            "scene_id": meta.get("scene_id"),
            "category": meta.get("category"),
            "gold_layer_ids": meta.get("gold_layer_ids"),
            "distractor_layer_ids": meta.get("distractor_layer_ids"),
            "demand_level": meta.get("demand_level"),
        }
