#!/usr/bin/env python3
"""Re-grade finished conversations with an arbitrary judge model.

Two things this adds over the grading that ran inside the simulators.

**Cross-judging.** Every conversation in the original runs was graded by the
model that produced it (`assistant_model == judge_model` in 100% of them), so a
model that is lenient about its own output is indistinguishable from a model
that is good. This script re-grades any assistant's logs with any judge,
including the assistant itself, so the full judge x assistant matrix can be
filled in and self-preference measured as the diagonal's excess over what judge
leniency and assistant ability predict.

**Premature commitment.** Simulator grading answered one question: did the
reference answer appear *anywhere* in the transcript. That hides the thing the
sharded conditions exist to measure — whether the model committed to the right
answer before the evidence supporting it had been delivered. The judge here also
reports which assistant turn first carried the answer, and this script pairs that
turn with the reveal log to work out what was still withheld at that moment:

    final_turn_correct   the answer is in the last assistant turn
    premature            the answer first appeared before the last turn
    premature_incomplete premature *and* shards were still unrevealed
    withheld_visual      of those unrevealed shards, at least one carried a
                         layer image -- the model answered without seeing
                         evidence it was going to be given
    withheld_text_only   shards remained but none of them carried an image
    abandoned            answered correctly early, wrong in the final turn

The judge additionally lists *every* answer attempt, right or wrong, which is
what the position of the first attempt is measured from, and everything below is
then computed here without further model calls:

    first_attempt_bin    which fifth of the conversation the first attempt falls
                         in (0-20% .. 80-100%), so accuracy can be read against
                         how early the model committed to anything at all
    attempt_lengths      characters of each answering turn, in order, for the
                         answer-bloat curve
    response_chars       characters of every assistant turn, for the verbosity
                         quintiles computed per (model, condition)
    gold_revealed_at_*   whether the gold layers this question needs had actually
                         been delivered when the model answered
    ungrounded_correct   correct while at least one gold layer was still
                         unrevealed -- a right answer the evidence did not yet
                         support
    layer_mentions       per turn, which layers the turn names by their
                         annotated label, and which turn introduced each layer

`ungrounded_correct` is the one that changes how the headline numbers should be
read. A benchmark that scores "the answer appeared somewhere" credits a model
equally for deducing an answer from the layer it was given and for guessing it
from the question's phrasing before any supporting layer arrived. The two are
separable here only because the dataset annotates `gold_layer_ids` per question
and the reveal log says when each layer landed.

A single-turn condition (FULL, CONCAT) has one assistant turn and everything
revealed at once, so it reports premature=False by construction. That is correct,
not a gap: there is no early turn to commit in.

Grading is text-only. The checkpoints are VLMs, but the judge reads a rendered
transcript with image parts replaced by a bare `[image]` -- no pixels reach it,
and no image path does either, since a path like `.../layer_2.png` can name the
answer.

Usage:
    python posthoc_judge.py --log_folder logs/Final-0 --judge_model qwen3.5-9b
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import tqdm

from multimodal import content_to_text
from tasks.layered_vqa.judge import normalize, shared_judge_cache

PROMPT_PATH = Path("prompts/layered_vqa/layered_vqa_posthoc_judge.txt")
DEFAULT_DATA_DIR = Path("../../data/derived/lic")

# Both simulators mark a reveal, under different names: `sharded` and
# `recap-concat` log `shard_revealed`, `snowball` logs `hint_revealed`. Missing
# entirely on single-turn conv_types, where nothing is progressive.
REVEAL_TYPES = {"shard_revealed", "hint_revealed"}

ANSWER_DESCRIPTION = (
    "The answer should be a short phrase naming an object, an attribute, "
    "a count, or a spatial relation (for example 'blue', 'a computer mouse', "
    "'two', 'to the left of the chair')."
)

# Conversation position is reported in fifths, the same five bins the
# lost-in-conversation analysis uses, so the two are directly comparable.
PROGRESS_BINS = (
    (0.2, "0-20%"),
    (0.4, "20-40%"),
    (0.6, "40-60%"),
    (0.8, "60-80%"),
    (1.0, "80-100%"),
)

_print_lock = threading.Lock()


def repair_surrogates(text: str) -> str:
    """Make a string UTF-8 encodable, keeping whatever meaning survives.

    Models emit lone surrogates -- half of an emoji pair, U+D83D with no
    partner. Python holds one happily and then raises when the string is
    written to a UTF-8 file, which is not something json.dumps or the file
    object can be told to tolerate after the fact.

    The utf-16 surrogatepass round-trip fixes both shapes at once: a genuine
    pair reassembles into the emoji it was meant to be, and an unpaired half
    degrades to U+FFFD. The fast path is a plain encode, so clean text -- which
    is almost all of it -- costs one encode and no allocation.
    """
    try:
        text.encode("utf-8")
        return text
    except UnicodeEncodeError:
        return text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")


def as_text(value: Any) -> str:
    """Coerce a judge-supplied field to a string.

    The JSON parses and every key is present, but the *types* are not
    guaranteed: Pixtral returns `answer_span` as a list of strings often enough
    to matter, and a list has no .strip(). Joining rather than rejecting keeps
    the citation usable -- a multi-part span still locates its turn -- while a
    number or a null degrades to its text form or to empty.
    """
    if isinstance(value, str):
        return repair_surrogates(value)
    if isinstance(value, (list, tuple)):
        return " ".join(as_text(v) for v in value if v is not None)
    if value is None or isinstance(value, (dict, bool)):
        return ""
    return repair_surrogates(str(value))


def log(message: str) -> None:
    """Print above the progress bar rather than through it.

    `tqdm.write` is what keeps a message from being interleaved into the bar's
    own line; a bare print would leave a smeared bar behind every log line.
    Same idiom the simulators use.
    """
    with _print_lock:
        tqdm.tqdm.write(message)


# ---------------------------------------------------------------------------
# transcript
# ---------------------------------------------------------------------------

def render_numbered(trace: list[dict[str, Any]]) -> tuple[str, int]:
    """Render the conversation with assistant turns numbered, for the judge.

    Numbering is what lets the judge name a turn instead of only quoting one,
    and it is done here rather than by the judge counting for itself, so the
    numbers in its answer refer to the same turns this script indexes.
    """
    lines: list[str] = []
    n_assistant = 0
    for message in trace:
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        text = content_to_text(message.get("content"), image_placeholder="[image]")
        if role == "assistant":
            n_assistant += 1
            lines.append(f"[assistant #{n_assistant}] {text}")
        else:
            lines.append(f"[user] {text}")
    return "\n\n".join(lines), n_assistant


def assistant_texts(trace: list[dict[str, Any]]) -> list[str]:
    return [
        content_to_text(m.get("content"), image_placeholder="[image]")
        for m in trace
        if m.get("role") == "assistant"
    ]


def revealed_by_turn(trace: list[dict[str, Any]]) -> list[set[int]]:
    """Shard ids revealed at or before each assistant turn, in turn order.

    Walks the trace once, accumulating reveal log entries, and snapshots the
    accumulator at every assistant message. Index i holds what the model could
    have known when it produced assistant turn i+1.
    """
    seen: set[int] = set()
    snapshots: list[set[int]] = []
    for message in trace:
        if message.get("role") == "log":
            content = message.get("content") or {}
            if content.get("type") in REVEAL_TYPES:
                shard_id = content.get("shard_id")
                if isinstance(shard_id, int):
                    seen.add(shard_id)
        elif message.get("role") == "assistant":
            snapshots.append(set(seen))
    return snapshots


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------

class Datasets:
    """Sample lookup by (dataset_fn, task_id), loaded once per file."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._cache: dict[str, dict[str, dict]] = {}
        self._lock = threading.Lock()

    def get(self, dataset_fn: str, task_id: str) -> dict | None:
        name = os.path.basename(dataset_fn or "")
        with self._lock:
            table = self._cache.get(name)
            if table is None:
                path = self.data_dir / name
                if not path.exists():
                    self._cache[name] = {}
                    return None
                rows = json.loads(path.read_text(encoding="utf-8"))
                table = {r["task_id"]: r for r in rows}
                self._cache[name] = table
        return table.get(task_id)


def shard_image_map(sample: dict) -> dict[int, bool]:
    """shard_id -> whether that shard attaches at least one layer image."""
    out: dict[int, bool] = {}
    for shard in sample.get("shards", []) or []:
        out[int(shard["shard_id"])] = bool(shard.get("images"))
    return out


_LAYER_RE = re.compile(r"layer_(\d+)\.png")


def shard_layer_map(sample: dict) -> dict[int, list[int]]:
    """shard_id -> the layer ids that shard attaches.

    Layer identity lives only in the render filename, which is also why those
    paths must never reach the judge: `layer_2.png` next to a label of "bulb"
    would hand over the answer.
    """
    out: dict[int, list[int]] = {}
    for shard in sample.get("shards", []) or []:
        ids = []
        for image in shard.get("images") or []:
            match = _LAYER_RE.search(image.get("path", ""))
            if match:
                ids.append(int(match.group(1)))
        out[int(shard["shard_id"])] = ids
    return out


def layer_label_mentions(text: str, labels: dict[str, str]) -> list[int]:
    """Which layers a turn names, matched on the annotated layer label.

    Deterministic stand-in for a citation: the models are never asked to cite
    layer ids outside the grounding condition, so the label ("white cat",
    "bulb") is the only handle on which layer a sentence is talking about.
    Matched on normalized whole-token sequences to avoid "cat" firing inside
    "catch". It over-fires when two layers share a label and under-fires when a
    turn uses a synonym, so it is a distribution over turns, not a per-turn
    fact -- which is exactly how the citation-recency matrix reads it.
    """
    haystack = f" {normalize(text)} "
    found = []
    for raw_id, label in (labels or {}).items():
        needle = normalize(label)
        if needle and f" {needle} " in haystack:
            try:
                found.append(int(raw_id))
            except (TypeError, ValueError):
                continue
    return sorted(found)


def progress_bin(turn: int, n_turns: int) -> str:
    """Which fifth of the conversation a turn falls in.

    Position is measured as turn/n_turns, so the last turn is always 100% and
    lands in the final bin. A single-turn conversation is entirely "80-100%":
    there is no such thing as answering early when there is only one chance.
    """
    if not turn or not n_turns:
        return ""
    fraction = turn / n_turns
    for upper, name in PROGRESS_BINS[:-1]:
        if fraction <= upper + 1e-9:
            return name
    return PROGRESS_BINS[-1][1]


# ---------------------------------------------------------------------------
# judging
# ---------------------------------------------------------------------------

class PosthocJudge:
    """One judge model, one on-disk cache.

    The cache key includes the judge model, so the six passes over the same
    conversations never collide and a re-run after a crash costs nothing.
    """

    def __init__(self, model: str, cache_path: Path):
        from agent_config import agent_config

        self.model = model
        self.params = agent_config.judge()
        self.cache = shared_judge_cache(cache_path)
        self.prompt = PROMPT_PATH.read_text(encoding="utf-8")

    def key(self, task_id: str, transcript: str, reference: str) -> str:
        import hashlib

        payload = "\x1f".join(
            ["posthoc", self.model, task_id, normalize(transcript), normalize(reference)]
        )
        return "ph:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def __call__(
        self, transcript: str, reference: str, question: str, task_id: str, n_turns: int
    ) -> dict:
        if not transcript.strip() or n_turns == 0:
            # Nothing to grade. Prompting a judge with an empty conversation
            # invites it to invent one, so this verdict is not the LLM's.
            return {
                "is_correct": False,
                "first_correct_turn": 0,
                "final_turn_correct": False,
                "answer_span": "",
                "answer_attempts": [],
                "grader": "deterministic",
            }

        cache_key = self.key(task_id, transcript, reference)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return {
                "is_correct": cached["is_correct"],
                "first_correct_turn": cached.get("first_correct_turn", 0),
                "final_turn_correct": cached.get("final_turn_correct", False),
                "answer_span": as_text(cached.get("answer_span")),
                "reasoning": as_text(cached.get("reasoning")),
                "answer_attempts": cached.get("answer_attempts", []),
                "grader": "cache",
            }

        from model_openai import generate_json

        populated = (
            self.prompt.replace("[[QUESTION]]", question)
            .replace("[[REFERENCE_ANSWER]]", reference)
            .replace("[[ANSWER_DESCRIPTION]]", ANSWER_DESCRIPTION)
            .replace("[[CONVERSATION]]", transcript)
            .replace("[[N_TURNS]]", str(n_turns))
        )
        try:
            response = generate_json(
                [{"role": "user", "content": populated}],
                model=self.model,
                return_metadata=True,
                **self.params.call_kwargs(self.model),
            )
            payload = response["message"]
            is_correct = bool(payload.get("is_correct", False))
            try:
                first_turn = int(payload.get("first_correct_turn", 0) or 0)
            except (TypeError, ValueError):
                first_turn = 0
            first_turn = max(0, min(first_turn, n_turns))
            final_correct = bool(payload.get("final_turn_correct", False))
            answer_span = as_text(payload.get("answer_span"))
            reasoning = as_text(payload.get("reasoning"))
            attempts = clean_attempts(payload.get("answer_attempts"), n_turns)
        except Exception as error:
            # A judge outage is a failed grading, not a wrong answer: it is
            # written with grader="error" and no verdict is cached, so the next
            # launch retries it instead of freezing a false negative.
            return {
                "is_correct": False,
                "first_correct_turn": 0,
                "final_turn_correct": False,
                "answer_span": "",
                "answer_attempts": [],
                "grader": "error",
                "error": f"{type(error).__name__}: {error}",
            }

        record = {
            "first_correct_turn": first_turn,
            "final_turn_correct": final_correct,
            "answer_span": answer_span,
            "reasoning": reasoning,
            "answer_attempts": attempts,
            "task_id": task_id,
        }
        self.cache.put(cache_key, is_correct, record)
        return {"is_correct": is_correct, "grader": "llm", **record}


def clean_attempts(raw: Any, n_turns: int) -> list[dict]:
    """Normalise the judge's attempt list: in-range turns, one per turn, ordered.

    Models return this field loosely -- a turn number past the end, the same
    turn twice, a bare string instead of an object. Salvaging what parses beats
    discarding the call, but nothing is invented: an unusable entry is dropped,
    and a conversation whose whole list is unusable simply has no attempt data
    and is excluded from the attempt-position analysis rather than defaulting to
    turn 1, which would manufacture a premature answer.
    """
    if not isinstance(raw, list):
        return []
    by_turn: dict[int, dict] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            turn = int(entry.get("turn", 0) or 0)
        except (TypeError, ValueError):
            continue
        if not 1 <= turn <= n_turns:
            continue
        by_turn[turn] = {
            "turn": turn,
            "answer": as_text(entry.get("answer"))[:200],
            "correct": bool(entry.get("correct", False)),
        }
    return [by_turn[k] for k in sorted(by_turn)]


def locate_span(span: Any, turns: list[str]) -> int:
    """1-based index of the assistant turn containing `span`, else 0.

    An independent check on the judge's own `first_correct_turn`: the prompt
    demands a verbatim span, so the span should be findable in the turn the
    judge named. Exact match first, then a normalized match, which absorbs the
    whitespace and punctuation drift models introduce while copying.
    """
    span = as_text(span).strip()
    if not span:
        return 0
    for i, text in enumerate(turns, start=1):
        if span in text:
            return i
    needle = normalize(span)
    if not needle:
        return 0
    for i, text in enumerate(turns, start=1):
        if needle in normalize(text):
            return i
    return 0


# ---------------------------------------------------------------------------
# per-conversation record
# ---------------------------------------------------------------------------

def build_record(rec: dict, verdict: dict, sample: dict | None) -> dict:
    trace = rec["trace"]
    turns = assistant_texts(trace)
    n_turns = len(turns)
    snapshots = revealed_by_turn(trace)

    first_turn = int(verdict.get("first_correct_turn", 0) or 0)
    is_correct = bool(verdict.get("is_correct", False))
    span_turn = locate_span(verdict.get("answer_span", ""), turns)

    # The judge can mark a conversation correct and still fail to name a turn.
    # The verbatim span is the fallback, and both are recorded so the rate of
    # disagreement stays measurable rather than being silently papered over.
    resolved_turn = first_turn or span_turn
    if is_correct and resolved_turn == 0:
        resolved_turn = 0  # correct but unlocatable; counted in turn_unresolved

    final_turn_correct = bool(verdict.get("final_turn_correct", False))
    premature = bool(is_correct and 0 < resolved_turn < n_turns)

    total_shards = len(sample.get("shards", [])) if sample else 0
    has_image = shard_image_map(sample) if sample else {}

    revealed: set[int] = set()
    if resolved_turn and resolved_turn <= len(snapshots):
        revealed = snapshots[resolved_turn - 1]
    remaining = (
        {sid for sid in has_image if sid not in revealed} if has_image else set()
    )
    # Single-turn conditions log no reveals at all; everything arrived at once,
    # so "remaining" there is an artefact of the missing log, not withheld
    # evidence. Treat a conversation with no reveal markers as fully revealed.
    if not snapshots or not any(snapshots):
        remaining = set()

    remaining_visual = {s for s in remaining if has_image.get(s)}

    # ---- answer attempts, whether right or wrong --------------------------
    attempts = verdict.get("answer_attempts") or []
    attempt_turns = [a["turn"] for a in attempts]
    first_attempt_turn = attempt_turns[0] if attempt_turns else 0
    correct_turns = [a["turn"] for a in attempts if a["correct"]]
    # Distinct proposals, normalised, so "blue" and "Blue." are one answer and
    # a model that repeats itself is not scored as changing its mind.
    distinct = []
    for a in attempts:
        key = normalize(a["answer"])
        if key and key not in distinct:
            distinct.append(key)

    # ---- verbosity, from the transcript rather than the judge --------------
    # Character counts come from the logged turns, never from the judge's echo
    # of them: the judge is instructed to shorten what it quotes, so its text
    # would measure its own summarising rather than the assistant's verbosity.
    response_chars = [len(x) for x in turns]
    attempt_lengths = [
        {"attempt_index": i, "turn": a["turn"],
         "chars": len(turns[a["turn"] - 1]) if a["turn"] <= len(turns) else 0,
         "correct": a["correct"]}
        for i, a in enumerate(attempts, start=1)
    ]

    # ---- was the evidence actually there when the model answered? ---------
    gold = [int(x) for x in (sample or {}).get("meta", {}).get("gold_layer_ids", [])]
    shard_layers = shard_layer_map(sample) if sample else {}
    layer_shard = {lid: sid for sid, lids in shard_layers.items() for lid in lids}
    # Which assistant turn each layer first became visible at.
    layer_intro_turn: dict[int, int] = {}
    for turn_index, seen in enumerate(snapshots, start=1):
        for lid, sid in layer_shard.items():
            if sid in seen and lid not in layer_intro_turn:
                layer_intro_turn[lid] = turn_index

    def gold_ready(turn: int) -> bool:
        """Had every gold layer arrived by `turn`?

        True when the condition delivers no layers at all (the text arms), where
        there is no layer schedule to be early relative to; those conversations
        are excluded from the grounding split rather than counted as grounded.
        """
        if not gold or not layer_shard:
            return True
        return all(layer_intro_turn.get(g, 10**9) <= turn for g in gold)

    has_layer_schedule = bool(gold and layer_shard and any(snapshots))
    first_correct_grounded = gold_ready(resolved_turn) if resolved_turn else False
    ungrounded_correct = bool(
        is_correct and has_layer_schedule and resolved_turn and not first_correct_grounded
    )

    # ---- which layer each turn names, and when that layer arrived ----------
    labels = (sample or {}).get("meta", {}).get("layer_labels", {})
    mentions = [layer_label_mentions(x, labels) for x in turns] if labels else []

    return {
        "conv_id": rec.get("conv_id"),
        "task_id": rec.get("task_id"),
        "conv_type": rec.get("conv_type"),
        "dataset_fn": rec.get("dataset_fn"),
        "assistant_model": rec.get("assistant_model"),
        "judge_model": verdict.get("judge_model"),
        "original_is_correct": bool(rec.get("is_correct", False)),
        "is_correct": is_correct,
        "grader": verdict.get("grader"),
        "n_assistant_turns": n_turns,
        "first_correct_turn": first_turn,
        "span_turn": span_turn,
        "resolved_turn": resolved_turn,
        # The judge named a turn and the verbatim span was found in a different
        # one. Neither is authoritative; the flag makes the disagreement
        # countable so a judge that cannot cite itself can be excluded.
        "turn_conflict": bool(first_turn and span_turn and first_turn != span_turn),
        "turn_unresolved": bool(is_correct and resolved_turn == 0),
        "final_turn_correct": final_turn_correct,
        "premature": premature,
        "premature_incomplete": bool(premature and remaining),
        "withheld_visual": bool(premature and remaining_visual),
        "withheld_text_only": bool(premature and remaining and not remaining_visual),
        "abandoned": bool(premature and not final_turn_correct),
        "shards_total": total_shards,
        "shards_revealed_at_answer": len(revealed),
        "shards_remaining_at_answer": len(remaining),
        "visual_shards_remaining_at_answer": len(remaining_visual),
        # -- answer attempts -----------------------------------------------
        "n_attempts": len(attempts),
        "attempt_turns": attempt_turns,
        "first_attempt_turn": first_attempt_turn,
        "first_attempt_progress": (first_attempt_turn / n_turns) if (first_attempt_turn and n_turns) else None,
        "first_attempt_bin": progress_bin(first_attempt_turn, n_turns),
        "first_attempt_correct": bool(attempts and attempts[0]["correct"]),
        "last_correct_turn": correct_turns[-1] if correct_turns else 0,
        "n_distinct_answers": len(distinct),
        # Changed its proposed answer at least once. Paired with is_correct this
        # separates a model that reconsidered into the right answer from one
        # that reconsidered out of it (`abandoned`).
        "changed_answer": len(distinct) > 1,
        "answer_attempts": attempts,
        # -- verbosity -------------------------------------------------------
        "response_chars": response_chars,
        "total_chars": sum(response_chars),
        "mean_response_chars": (sum(response_chars) / len(response_chars)) if response_chars else 0,
        "final_response_chars": response_chars[-1] if response_chars else 0,
        "attempt_lengths": attempt_lengths,
        # -- evidence grounding ----------------------------------------------
        "gold_layer_ids": gold,
        "layer_intro_turn": {str(k): v for k, v in sorted(layer_intro_turn.items())},
        "has_layer_schedule": has_layer_schedule,
        "first_correct_grounded": bool(is_correct and first_correct_grounded),
        # Right answer, but at a turn when a gold layer had not yet been
        # delivered. Not a bug in the run and not necessarily luck -- but it is
        # not evidence-following either, and the headline accuracy cannot tell
        # the two apart.
        "ungrounded_correct": ungrounded_correct,
        "gold_layers_revealed_at_answer": sum(
            1 for g in gold if layer_intro_turn.get(g, 10**9) <= (resolved_turn or 0)
        ),
        "gold_layers_total": len(gold),
        # -- layer citations --------------------------------------------------
        "layer_mentions": mentions,
        "answer_span": verdict.get("answer_span", ""),
        "reasoning": verdict.get("reasoning", ""),
        "error": verdict.get("error", ""),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def conversation_files(log_folder: Path, conv_types: list[str] | None) -> list[Path]:
    base = log_folder / "layered_vqa"
    if not base.is_dir():
        return []
    files = []
    for path in sorted(base.rglob("*.jsonl")):
        if conv_types and path.parent.name not in conv_types:
            continue
        files.append(path)
    return files


def pending_in(path: Path, out_path: Path, assistants: list[str] | None,
               limit: int) -> tuple[list[dict], int]:
    """Records still to grade in one file, and how many were already done.

    Returns the parsed records rather than a count so the caller does not read
    and parse the file twice. Memory stays bounded because this is called one
    file at a time, but it does mean the bar's total is only known after the
    whole pass has been planned -- hence `plan()` below.
    """
    done = already_done(out_path)
    records: list[dict] = []
    skipped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if assistants and rec.get("assistant_model") not in assistants:
            continue
        if rec.get("conv_id") in done:
            skipped += 1
            continue
        records.append(rec)
        if limit and len(records) >= limit:
            break
    return records, skipped


def already_done(out_path: Path) -> set[str]:
    """conv_ids already graded, so an interrupted pass resumes where it left off.

    Only real verdicts count. A judge outage is a failed grading, not a wrong
    answer, so errored conversations are kept out of this file entirely (they go
    to errors.jsonl) and the next launch regrades them. Treating an error as
    done would freeze a false negative into the results.
    """
    if not out_path.exists():
        return set()
    done = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("grader") != "error" and "conv_id" in record:
            done.add(record["conv_id"])
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--log_folder", required=True,
                        help="A run folder, e.g. logs/Final-0")
    parser.add_argument("--judge_model", required=True,
                        help="models.yaml key of the grading model")
    parser.add_argument("--assistants", nargs="*", default=None,
                        help="Only re-grade these assistant models (default: all found)")
    parser.add_argument("--conv_types", nargs="*", default=None,
                        help="Only these conv_type folders (default: all)")
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--subdir", default="posthoc",
                        help="Directory under the log folder to write into. "
                             "The default keeps the cross-judge matrix's own "
                             "output where run_posthoc_judging.sh and every "
                             "existing logs/Final-*/posthoc/ tree already have "
                             "it; the closed-model runners pass "
                             "'primary-analysis', because a measure produced "
                             "by the primary run should not be filed under a "
                             "name that says it came later")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after N conversations per file (smoke tests)")
    args = parser.parse_args()

    log_folder = Path(args.log_folder)
    if not (log_folder / "layered_vqa").is_dir():
        print(f"No conversation logs under {log_folder}/layered_vqa", file=sys.stderr)
        return 1

    out_root = log_folder / args.subdir / args.judge_model
    out_root.mkdir(parents=True, exist_ok=True)
    judge = PosthocJudge(args.judge_model, out_root / "judge_cache.jsonl")
    datasets = Datasets(Path(args.data_dir))

    files = conversation_files(log_folder, args.conv_types)
    if not files:
        print(f"No .jsonl conversation files under {log_folder}", file=sys.stderr)
        return 1

    totals = Counter()
    write_lock = threading.Lock()

    # Plan the whole pass before grading any of it. Counting first costs one
    # extra parse of the conversation logs, and buys a bar with a real total and
    # therefore a real ETA -- worth it when the alternative is hours of silence
    # punctuated by one line per file. Only the records are held, one file at a
    # time, so peak memory is a single file's worth of traces.
    plan: list[tuple[Path, Path, list[dict]]] = []
    for path in files:
        out_dir = out_root / path.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / path.name
        records, skipped = pending_in(path, out_path, args.assistants, args.limit)
        totals["skipped"] += skipped
        if records:
            plan.append((path, out_path, records))

    pending = sum(len(r) for _, _, r in plan)
    if not pending:
        log(f"[{args.judge_model}] {log_folder.name}: nothing to do "
            f"({totals['skipped']} already graded)")
        return 0

    log(f"[{args.judge_model}] {log_folder.name}: {pending} conversations to grade "
        f"across {len(plan)} files ({totals['skipped']} already done)")

    # errors="replace" is the backstop for the same class of failure: an
    # unencodable character reaching the write would otherwise raise inside the
    # lock, past every guard, and kill a pass holding hours of work.
    error_handle = (out_root / "errors.jsonl").open(
        "a", encoding="utf-8", errors="replace")
    bar = tqdm.tqdm(total=pending, unit="conv",
                    desc=f"{args.judge_model} -> {log_folder.name}")

    for path, out_path, records in plan:
        handle = out_path.open("a", encoding="utf-8", errors="replace")

        def grade(rec: dict) -> None:
            # Nothing about one conversation may end the pass. `PosthocJudge`
            # already contains transport and parse failures, but a reply that
            # parses and is merely the wrong *shape* used to raise out here in
            # build_record and kill hours of completed grading -- while the
            # offending verdict sat in the cache, so the next launch replayed
            # the same crash. Anything unexpected is now recorded as an error
            # and retried on the next run, exactly like a judge outage.
            try:
                sample = datasets.get(rec.get("dataset_fn", ""), rec.get("task_id", ""))
                question = (sample or {}).get("question", "")
                reference = (sample or {}).get("answer", "")
                transcript, n_turns = render_numbered(rec["trace"])
                verdict = judge(transcript, reference, question, rec.get("task_id", ""), n_turns)
                verdict["judge_model"] = args.judge_model
                out = build_record(rec, verdict, sample)
            except Exception as error:
                out = {
                    "conv_id": rec.get("conv_id"),
                    "task_id": rec.get("task_id"),
                    "conv_type": rec.get("conv_type"),
                    "dataset_fn": rec.get("dataset_fn"),
                    "assistant_model": rec.get("assistant_model"),
                    "judge_model": args.judge_model,
                    "is_correct": False,
                    "grader": "error",
                    "premature": False,
                    "premature_incomplete": False,
                    "withheld_visual": False,
                    "abandoned": False,
                    "turn_unresolved": False,
                    "turn_conflict": False,
                    "ungrounded_correct": False,
                    "changed_answer": False,
                    "n_attempts": 0,
                    "error": f"{type(error).__name__}: {error}",
                }
            payload = json.dumps(out, ensure_ascii=False)
            # Errors go to their own file, so the results file holds only real
            # verdicts and the next launch retries the failures.
            target = error_handle if out["grader"] == "error" else handle
            with write_lock:
                target.write(payload + "\n")
                target.flush()
                totals["graded"] += 1
                totals[out["grader"]] += 1
                if out["is_correct"]:
                    totals["correct"] += 1
                if out["premature"]:
                    totals["premature"] += 1
                if out["premature_incomplete"]:
                    totals["premature_incomplete"] += 1
                if out["withheld_visual"]:
                    totals["withheld_visual"] += 1
                if out["abandoned"]:
                    totals["abandoned"] += 1
                if out["turn_unresolved"]:
                    totals["turn_unresolved"] += 1
                if out["turn_conflict"]:
                    totals["turn_conflict"] += 1
                if out["ungrounded_correct"]:
                    totals["ungrounded_correct"] += 1
                if out["changed_answer"]:
                    totals["changed_answer"] += 1
                if out["n_attempts"] == 0:
                    totals["no_attempts_reported"] += 1
                bar.update(1)
                if totals["error"]:
                    bar.set_postfix(err=totals["error"], refresh=False)

        try:
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                list(pool.map(grade, records))
        finally:
            handle.close()

    bar.close()
    error_handle.close()

    graded = totals["graded"] or 1
    log(
        f"\n[{args.judge_model}] {log_folder.name}: graded {totals['graded']} "
        f"(skipped {totals['skipped']} already done, {totals['error']} judge errors "
        f"-> posthoc/{args.judge_model}/errors.jsonl, retried next run)\n"
        f"  correct              {totals['correct']} ({totals['correct']/graded:.3f})\n"
        f"  premature            {totals['premature']} ({totals['premature']/graded:.3f})\n"
        f"  premature_incomplete {totals['premature_incomplete']}\n"
        f"  withheld_visual      {totals['withheld_visual']}\n"
        f"  abandoned            {totals['abandoned']}\n"
        f"  ungrounded_correct   {totals['ungrounded_correct']}\n"
        f"  changed_answer       {totals['changed_answer']}\n"
        f"  turn_unresolved      {totals['turn_unresolved']}   "
        f"turn_conflict {totals['turn_conflict']}   "
        f"no_attempts {totals['no_attempts_reported']}"
    )
    return 1 if totals["error"] and not totals["graded"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
