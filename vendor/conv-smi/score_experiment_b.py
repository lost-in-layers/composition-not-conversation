"""Aggregate the Experiment B logs into one table per condition.

Experiment B is 14 conditions that differ in *how* the question and the visual
evidence are delivered, not in what is asked. Each one is identified by the pair
(conv_type folder, dataset file) — several conditions share a folder and are
told apart only by the dataset, so both halves are load-bearing:

    SHARDED_V   sharded-scripted + layered_vqa_visual_canonical.json
    ALIGNED_VQ  sharded-scripted + layered_vqa_aligned_vq.json

Accuracy comes from the logged judge verdict, the same field
`score_experiment_a.py` reads. This script adds one thing Experiment A does not
need: a **coverage** column. A conversation that overflowed the server's context
window raises rather than logging a wrong answer, so the sample is simply absent
from the file — an accuracy computed over what survived would quietly be an
accuracy over the easy subset. Coverage is logged conversations / dataset size,
and anything below 1.000 is called out.

Run it from this directory, like everything else here:

    python score_experiment_b.py --log_folder logs/Layered-VQA-0
    python score_experiment_b.py --models qwen3.5-9b --by_model
    python score_experiment_b.py --csv exp_b.csv    # one row per conversation
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, NamedTuple

from score_experiment_a import (
    evaluation_of,
    fmt,
    group_by,
    load_sample_meta,
    mean,
    render_table,
    write_csv,
)


class Condition(NamedTuple):
    """One Experiment-B arm: a label, the log folder, and the dataset it ran on."""

    label: str
    conv_type: str
    dataset_fn: str
    group: str


# Order and labels follow README-EXP section "Experiment B". FULL is condition 1
# there and doubles as Experiment A's A1, which is why it logs under the same
# conv_type/dataset pair as A1 and resumes across both scripts.
CONDITIONS: tuple[Condition, ...] = (
    Condition("1 FULL", "full", "layered_vqa_text.json", "baseline"),
    Condition("2 CONCAT_Q", "concat", "layered_vqa_text.json", "text"),
    Condition("3 SHARDED_Q", "sharded", "layered_vqa_text.json", "text"),
    Condition("4 RECAP_Q", "recap-concat", "layered_vqa_text.json", "text"),
    Condition("5 SNOWBALL_Q", "snowball", "layered_vqa_text.json", "text"),
    Condition("6 CONCAT_V", "full", "layered_vqa_visual_canonical.json", "visual"),
    Condition("7 SHARDED_V", "sharded-scripted", "layered_vqa_visual_canonical.json", "visual"),
    Condition("8 RECAP_V", "recap-concat", "layered_vqa_visual_canonical.json", "visual"),
    Condition("9 SNOWBALL_V", "snowball-scripted", "layered_vqa_visual_canonical.json", "visual"),
    Condition("10 SHARDED_VQ aligned", "sharded-scripted", "layered_vqa_aligned_vq.json", "main"),
    Condition("11 SHARDED_VQ misaligned", "sharded-scripted", "layered_vqa_misaligned_vq.json", "main"),
    Condition("12 EVIDENCE_FIRST_V", "sharded-scripted", "layered_vqa_visual_evidence_first.json", "additional"),
    Condition("13 EVIDENCE_LAST_V", "sharded-scripted", "layered_vqa_visual_evidence_last.json", "additional"),
    Condition("14 SHUFFLE_Q-CONCAT", "shuffle-concat", "layered_vqa_text.json", "additional"),
)

BASELINE_LABEL = CONDITIONS[0].label

# Contrasts the paper reports as a pair. Each is (label, condition, reference);
# the number printed is condition minus reference, in accuracy points.
CONTRASTS = (
    ("Sharding cost (text)", "3 SHARDED_Q", "2 CONCAT_Q"),
    ("Recap recovery (text)", "4 RECAP_Q", "3 SHARDED_Q"),
    ("Sharding cost (visual)", "7 SHARDED_V", "6 CONCAT_V"),
    ("Recap recovery (visual)", "8 RECAP_V", "7 SHARDED_V"),
    ("Alignment cost", "11 SHARDED_VQ misaligned", "10 SHARDED_VQ aligned"),
    ("Evidence order cost", "13 EVIDENCE_LAST_V", "12 EVIDENCE_FIRST_V"),
    ("Shard order cost (text)", "14 SHUFFLE_Q-CONCAT", "2 CONCAT_Q"),
)


def matching_folders(root: Path, conv_type: str) -> list[Path]:
    """Every log folder for a conv_type, including its temperature variants.

    `utils_log.temperature_suffix` appends `-t<x>` or `-at<x>-ut<y>` when a run
    used a non-baseline temperature, so an exact name match would silently miss
    those runs. The suffix check also keeps `sharded` from swallowing
    `sharded-scripted`.
    """
    if not root.is_dir():
        return []
    folders = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if name == conv_type or name.startswith(f"{conv_type}-t") or name.startswith(f"{conv_type}-at"):
            folders.append(entry)
    return folders


def dataset_size(dataset_dir: Path, dataset_fn: str) -> int | None:
    path = dataset_dir / os.path.basename(dataset_fn)
    if not path.exists():
        return None
    return len(json.loads(path.read_text(encoding="utf-8")))


def turn_count(record: dict[str, Any]) -> int:
    """Assistant turns in the conversation — the multi-turn conditions' length."""
    return sum(1 for entry in record.get("trace", []) if entry.get("role") == "assistant")


def collect_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    from utils_log import load_results_from

    root = Path(args.log_folder) / args.task
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        meta_by_task = load_sample_meta(Path(args.dataset_dir), condition.dataset_fn)
        for folder in matching_folders(root, condition.conv_type):
            # merge_trapi=False for the same reason score_experiment_a.py wants
            # it: the model name here is the models.yaml alias that --models
            # filters on, and rewriting it makes the filter match nothing.
            by_model = load_results_from(str(folder), condition.dataset_fn, merge_trapi=False)
            for model, records in by_model.items():
                if args.models and model not in args.models:
                    continue
                for record in records:
                    meta = meta_by_task.get(record["task_id"], {})
                    evaluation = evaluation_of(record)
                    rows.append(
                        {
                            "condition": condition.label,
                            "group": condition.group,
                            "conv_type": folder.name,
                            "dataset_fn": os.path.basename(condition.dataset_fn),
                            "model": model,
                            "task_id": record["task_id"],
                            "category": meta.get("category"),
                            "demand_level": meta.get("demand_level"),
                            "shard_count": meta.get("shard_count"),
                            "is_correct": bool(record.get("is_correct")),
                            "assistant_turns": turn_count(record),
                            "exact_answer": evaluation.get("answer_span"),
                        }
                    )
    return rows


def summarize(rows: list[dict[str, Any]], expected: int | None) -> dict[str, Any]:
    questions = {row["task_id"] for row in rows}
    return {
        "n": len(rows),
        "questions": len(questions),
        "coverage": len(questions) / expected if expected else None,
        "answer_acc": mean(row["is_correct"] for row in rows),
        "turns": mean(row["assistant_turns"] for row in rows),
    }


OVERVIEW_COLUMNS = [
    ("n", "n"),
    ("questions", "questions"),
    ("coverage", "coverage"),
    ("answer_acc", "answer acc"),
    ("turns", "assistant turns"),
]


def report(rows: list[dict[str, Any]], by_model: bool, sizes: dict[str, int | None]) -> str:
    out: list[str] = ["# Experiment B", ""]

    present = {row["condition"] for row in rows}
    missing = [c.label for c in CONDITIONS if c.label not in present]
    if missing:
        out += [f"*Not yet run: {', '.join(missing)}.*", ""]

    by_condition = group_by(rows, "condition")
    order = {condition.label: i for i, condition in enumerate(CONDITIONS)}
    headers = ["condition"] + (["model"] if by_model else []) + [
        label for _, label in OVERVIEW_COLUMNS
    ]
    table_rows: list[list[Any]] = []
    incomplete: list[str] = []
    for label in sorted(by_condition, key=lambda x: order.get(x, len(order))):
        condition_rows = by_condition[label]
        groups = (
            sorted(group_by(condition_rows, "model").items())
            if by_model
            else [(None, condition_rows)]
        )
        for model, group in groups:
            summary = summarize(group, sizes.get(label))
            prefix = [label] + ([model] if by_model else [])
            table_rows.append(prefix + [summary[key] for key, _ in OVERVIEW_COLUMNS])
            coverage = summary["coverage"]
            if coverage is not None and coverage < 1.0:
                who = f"{label} ({model})" if model else label
                incomplete.append(
                    f"{who}: {summary['questions']}/{sizes.get(label)} questions logged"
                )
    out += [render_table(headers, table_rows), ""]

    if incomplete:
        # A missing conversation is not a wrong answer; it is an accuracy
        # computed over a different, self-selected question set.
        out += [
            "**Incomplete conditions** — these accuracies are over a subset, "
            "not the full question set:",
            "",
            *(f"- {line}" for line in incomplete),
            "",
        ]

    out += contrast_section(rows, by_model)

    for key, title in (("category", "By category"), ("demand_level", "By demand level")):
        if not any(row[key] is not None for row in rows):
            continue
        out += [f"## {title}", ""]
        headers = [key] + [c.label for c in CONDITIONS if c.label in present]
        grouped = group_by(rows, key)
        table_rows = []
        for value in sorted(grouped, key=lambda v: (v is None, v)):
            by_label = group_by(grouped[value], "condition")
            table_rows.append(
                [value]
                + [
                    mean(row["is_correct"] for row in by_label[c.label])
                    if c.label in by_label
                    else None
                    for c in CONDITIONS
                    if c.label in present
                ]
            )
        out += [render_table(headers, table_rows), ""]

    return "\n".join(out)


def contrast_section(rows: list[dict[str, Any]], by_model: bool) -> list[str]:
    """The paired deltas the paper reads, computed over shared questions only.

    Restricting to questions both arms logged is what makes the delta a paired
    comparison; pooling would let a condition that dropped its longest
    conversations look better than the arm that answered them.
    """
    models = sorted({row["model"] for row in rows}) if by_model else [None]
    accuracy: dict[tuple[Any, str], dict[str, bool]] = {}
    for row in rows:
        key = (row["model"] if by_model else None, row["condition"])
        accuracy.setdefault(key, {})[row["task_id"]] = row["is_correct"]

    table_rows: list[list[Any]] = []
    for model in models:
        for label, condition, reference in CONTRASTS:
            left = accuracy.get((model, condition))
            right = accuracy.get((model, reference))
            if not left or not right:
                continue
            shared = sorted(set(left) & set(right))
            if not shared:
                continue
            left_acc = mean(left[task_id] for task_id in shared)
            right_acc = mean(right[task_id] for task_id in shared)
            row: list[Any] = [label]
            if by_model:
                row.append(model)
            table_rows.append(
                row
                + [
                    f"{condition} - {reference}",
                    len(shared),
                    right_acc,
                    left_acc,
                    left_acc - right_acc,
                ]
            )
    if not table_rows:
        return []

    headers = ["contrast"] + (["model"] if by_model else []) + [
        "arms",
        "paired n",
        "reference acc",
        "condition acc",
        "delta",
    ]
    return [
        "## Paired contrasts",
        "",
        "Computed over the questions both arms logged.",
        "",
        render_table(headers, table_rows),
        "",
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--log_folder", default="logs")
    parser.add_argument("--task", default="layered_vqa")
    parser.add_argument("--dataset_dir", default="../../data/derived/lic")
    parser.add_argument("--models", nargs="+", default=None, help="Only score these assistant models")
    parser.add_argument("--by_model", action="store_true", help="One row per condition and model instead of pooling models")
    parser.add_argument("--csv", default=None, help="Also write one row per conversation here")
    return parser.parse_args(argv)


def build_report(args: argparse.Namespace) -> tuple[str, list[dict[str, Any]]]:
    """The markdown report and the flat rows behind it, for reuse by callers."""
    rows = collect_rows(args)
    sizes = {
        condition.label: dataset_size(Path(args.dataset_dir), condition.dataset_fn)
        for condition in CONDITIONS
    }
    if not rows:
        return "# Experiment B\n\n*No conversations matched.*\n", rows
    return report(rows, by_model=args.by_model, sizes=sizes), rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    text, rows = build_report(args)
    if not rows:
        raise SystemExit("no conversations matched — check --log_folder and --models")
    print(text)
    if args.csv:
        write_csv(rows, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
