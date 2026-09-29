"""Reading and scoring the evidence a model claims to have used.

Experiment A condition 2 asks for an answer *and* the supporting layer ids, in
a labelled-line format (`Answer:` / `Layers:` / `Confidence:` / `Why:`). This
module turns such a reply into a prediction and scores that prediction against
the annotation. Everything here is deterministic and model-free: the LLM judge
decides the answer, these functions decide the grounding, and the two verdicts
are combined in `TaskLayeredVQA.evaluate_conversation`.

A reply that does not carry a readable `Layers:` line yields `layer_ids=None`,
not an empty list. The distinction matters: an empty set is a claim ("no layer
supported this"), a missing line is a format failure, and folding the second
into the first would report every non-compliant model as a confidently
ungrounded one.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

# Labels tolerate the markdown a chat model tends to add ("**Answer:** ...").
_LABEL_PREFIX = r"^[\s>*_#\-]*"
_ANSWER_RE = re.compile(_LABEL_PREFIX + r"answer\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)
_LAYERS_RE = re.compile(_LABEL_PREFIX + r"layers?\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)
_CONFIDENCE_RE = re.compile(_LABEL_PREFIX + r"confidence\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)
_WHY_RE = re.compile(_LABEL_PREFIX + r"(?:why|explanation)\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)

# Layer ids are non-negative, so the minus in "1-3" is a separator, not a sign:
# matching it would turn a sloppy range into the invalid id -3.
_INT_RE = re.compile(r"\d+")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
# An explicit "I used no layers", as opposed to an unreadable line.
_EMPTY_RE = re.compile(r"^(none|no|nil|n/?a|nothing|no\s+layers?|none\s+of\s+them)\b", re.IGNORECASE)


def _last_match(pattern: re.Pattern[str], text: str) -> str | None:
    """The last value for a label — models restate the block before finishing."""
    matches = pattern.findall(text or "")
    return matches[-1] if matches else None


def _clean(value: str) -> str:
    return value.strip().strip("*`").strip()


def _parse_confidence(raw: str | None) -> float | None:
    if raw is None:
        return None
    match = _NUMBER_RE.search(raw)
    if match is None:
        return None
    value = float(match.group())
    if "%" in raw or value > 1.0:
        # "80%" and a bare "80" both mean 0.8; anything above 100 is nonsense.
        value = value / 100.0
    if not 0.0 <= value <= 1.0:
        return None
    return value


def _parse_layer_ids(raw: str | None) -> tuple[list[int] | None, str | None]:
    """Layer ids from a `Layers:` value, or (None, reason) when unreadable."""
    if raw is None:
        return None, "no `Layers:` line in the reply"

    value = _clean(raw)
    ids = [int(match) for match in _INT_RE.findall(value)]
    if ids:
        seen: dict[int, None] = {}
        for layer_id in ids:
            seen.setdefault(layer_id, None)
        return list(seen), None

    if not value or _EMPTY_RE.match(value):
        return [], None
    return None, f"could not read layer ids from {value!r}"


def parse_layer_selection(reply: str) -> dict[str, Any]:
    """Read an `Answer:` / `Layers:` / `Confidence:` / `Why:` reply."""
    text = reply or ""
    answer_raw = _last_match(_ANSWER_RE, text)
    layers_raw = _last_match(_LAYERS_RE, text)
    confidence_raw = _last_match(_CONFIDENCE_RE, text)
    why_raw = _last_match(_WHY_RE, text)

    layer_ids, parse_error = _parse_layer_ids(layers_raw)

    return {
        "answer": _clean(answer_raw) if answer_raw is not None else None,
        "layer_ids": layer_ids,
        "confidence": _parse_confidence(confidence_raw),
        "confidence_raw": _clean(confidence_raw) if confidence_raw is not None else None,
        "explanation": _clean(why_raw) if why_raw is not None else "",
        # The layer set is what the condition is about, so the format is "ok"
        # exactly when the layers were readable.
        "format_ok": layer_ids is not None,
        "parse_error": parse_error,
    }


def _as_id_set(values: Iterable[Any] | None) -> set[int]:
    return {int(value) for value in (values or [])}


def layer_metrics(
    predicted: Sequence[int],
    *,
    gold: Iterable[Any],
    minimal_sets: Iterable[Iterable[Any]] | None = None,
    distractors: Iterable[Any] | None = None,
    candidates: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Score one predicted evidence set against a question's annotation."""
    pred_set = _as_id_set(predicted)
    gold_set = _as_id_set(gold)
    distractor_set = _as_id_set(distractors)
    candidate_set = _as_id_set(candidates)

    hits = pred_set & gold_set
    if not pred_set and not gold_set:
        precision = recall = f1 = 1.0
    else:
        precision = len(hits) / len(pred_set) if pred_set else 0.0
        recall = len(hits) / len(gold_set) if gold_set else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

    # An annotation with no minimal sets falls back to the gold set itself, so
    # the metric is always defined.
    accepted = [_as_id_set(entry) for entry in (minimal_sets or [])] or [gold_set]
    excess = pred_set - gold_set

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": pred_set == gold_set,
        "minimal_set_success": any(pred_set == entry for entry in accepted),
        "complete": gold_set <= pred_set,
        "n_predicted": len(pred_set),
        "n_gold": len(gold_set),
        "excess": len(excess),
        "excess_rate": len(excess) / len(pred_set) if pred_set else 0.0,
        "distractors_selected": len(pred_set & distractor_set),
        "distractor_captured": bool(pred_set & distractor_set),
        # Ids that name no layer of this scene. Cited ids outside the candidate
        # set are already counted as excess; listing them separately separates
        # "picked the wrong layer" from "invented a layer".
        "invalid_ids": sorted(pred_set - candidate_set) if candidate_set else [],
    }


def grounding_report(sample: dict[str, Any], reply: str) -> dict[str, Any]:
    """Parse a grounding reply and score it against the sample's annotation."""
    meta = sample.get("meta", {})
    selection = parse_layer_selection(reply)

    metrics = None
    if selection["layer_ids"] is not None:
        metrics = layer_metrics(
            selection["layer_ids"],
            gold=meta.get("gold_layer_ids", []),
            minimal_sets=meta.get("minimal_sufficient_layer_sets"),
            distractors=meta.get("distractor_layer_ids"),
            candidates=meta.get("candidate_layer_ids"),
        )

    return {"layer_selection": selection, "layer_metrics": metrics}
