"""Aggregate the Experiment A logs into the metrics table README-EXP asks for.

The three Experiment A conditions are all single-turn, so they all log under the
`full` conv_type and are told apart by the dataset they ran on:

    RGB Answer Only          layered_vqa_text.json
    All layers + selection   layered_vqa_grounding.json
    Oracle Gold Layers       layered_vqa_oracle_gold.json

Answer accuracy comes from the logged verdict; the grounding numbers come from
the `layer_metrics` block that `TaskLayeredVQA.evaluate_conversation` writes
into each conversation's `answer-evaluation` trace entry.

Run it from this directory, like everything else here:

    python score_experiment_a.py
    python score_experiment_a.py --datasets pilot_layered_vqa_text.json \\
        pilot_layered_vqa_grounding.json pilot_layered_vqa_oracle_gold.json
    python score_experiment_a.py --csv exp_a.csv     # one row per conversation

The CSV is the input for the paired tests and bootstrap CIs in README-EXP §5;
this script deliberately reports point estimates only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

# Which condition a dataset file measures. Matched on the file's basename with
# any `pilot_` prefix stripped, so pilot files score as their real condition.
CONDITIONS = {
    "layered_vqa_text.json": "A1 RGB answer only",
    "layered_vqa_grounding.json": "A2 All layers + selection",
    "layered_vqa_oracle_gold.json": "A3 Oracle gold layers",
}
DEFAULT_DATASETS = list(CONDITIONS)

GROUNDING_FIELDS = (
    "precision",
    "recall",
    "f1",
    "exact_match",
    "minimal_set_success",
    "complete",
    "distractor_captured",
    "excess",
    "n_predicted",
)


def condition_of(dataset_fn: str) -> str:
    basename = os.path.basename(dataset_fn)
    stripped = basename[len("pilot_"):] if basename.startswith("pilot_") else basename
    return CONDITIONS.get(stripped, basename)


def evaluation_of(record: dict[str, Any]) -> dict[str, Any]:
    """The judge's return value, as `grade_conversation` logged it."""
    for entry in reversed(record.get("trace", [])):
        content = entry.get("content")
        if entry.get("role") == "log" and isinstance(content, dict):
            if content.get("type") == "answer-evaluation":
                return content.get("evaluation_return") or {}
    return {}


def load_sample_meta(dataset_dir: Path, dataset_fn: str) -> dict[str, dict[str, Any]]:
    """task_id -> meta, for the category and demand-level breakdowns."""
    path = dataset_dir / os.path.basename(dataset_fn)
    if not path.exists():
        print(f"[warn] {path} not found; skipping its category/demand breakdowns")
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return {sample["task_id"]: sample.get("meta", {}) for sample in json.load(handle)}


def collect_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    """One flat row per logged conversation, across all three conditions."""
    from utils_log import load_results_from

    folder = os.path.join(args.log_folder, args.task, args.conv_type)
    if not os.path.isdir(folder):
        raise SystemExit(f"no logs at {folder} — run the conditions first")

    rows: list[dict[str, Any]] = []
    for dataset_fn in args.datasets:
        meta_by_task = load_sample_meta(Path(args.dataset_dir), dataset_fn)
        # merge_trapi rewrites the logged model name (`-instruct` and the
        # `t-`/`l-`/`b-` TrAPI prefixes are stripped), which is upstream's way
        # of merging one OpenAI model reached through two endpoints. Here the
        # name is the alias from `models.yaml` — the same string `--models`
        # takes and the report filename carries — so rewriting it made
        # `--models ministral3-8b-instruct` match nothing and print an empty
        # report.
        by_model = load_results_from(folder, dataset_fn, merge_trapi=False)
        for model, records in by_model.items():
            if args.models and model not in args.models:
                continue
            for record in records:
                evaluation = evaluation_of(record)
                selection = evaluation.get("layer_selection") or {}
                metrics = evaluation.get("layer_metrics") or {}
                meta = meta_by_task.get(record["task_id"], {})

                # Ids naming no layer of this scene: the model did not pick the
                # wrong evidence, it invented evidence. `excess` already counts
                # them, which buries an invented layer among ordinary
                # over-selection; kept separate they read as a distinct failure.
                # `None` outside A2, where no evidence set was ever asked for.
                invalid_ids = metrics.get("invalid_ids") if metrics else None

                row: dict[str, Any] = {
                    "condition": condition_of(dataset_fn),
                    "dataset_fn": os.path.basename(dataset_fn),
                    "model": model,
                    "task_id": record["task_id"],
                    "category": meta.get("category"),
                    "demand_level": meta.get("demand_level"),
                    "is_correct": bool(record.get("is_correct")),
                    "asks_for_layers": bool(evaluation.get("layer_selection") is not None),
                    "format_ok": selection.get("format_ok"),
                    "confidence": selection.get("confidence"),
                    "joint_correct": evaluation.get("joint_correct"),
                    "joint_complete": evaluation.get("joint_complete"),
                    "n_gold": metrics.get("n_gold"),
                    "invalid_citation": bool(invalid_ids) if metrics else None,
                    "n_invalid": len(invalid_ids or []) if metrics else None,
                }
                for field in GROUNDING_FIELDS:
                    row[field] = metrics.get(field)
                rows.append(row)
    return rows


# -- reporting ---------------------------------------------------------------


def mean(values: Iterable[Any]) -> float | None:
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(lines)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Point estimates for one group of conversations."""
    grounded = [row for row in rows if row["asks_for_layers"]]
    # Format failures have no evidence set, so they are excluded from the
    # per-set metrics and reported as their own rate. They still count against
    # joint accuracy: a reply nobody can read did not ground its answer.
    parsed = [row for row in grounded if row["format_ok"]]

    summary: dict[str, Any] = {
        "n": len(rows),
        "questions": len({row["task_id"] for row in rows}),
        "answer_acc": mean(row["is_correct"] for row in rows),
    }
    if grounded:
        summary.update(
            {
                "format_fail": mean(not row["format_ok"] for row in grounded),
                "joint_acc": mean(row["joint_correct"] for row in grounded),
                "joint_complete": mean(row["joint_complete"] for row in grounded),
                # Over `parsed`, like the other evidence-set metrics: a reply
                # with no readable `Layers:` line cited nothing, so it could
                # not have cited a layer that does not exist.
                "invalid_citation": mean(row["invalid_citation"] for row in parsed),
                **{field: mean(row[field] for row in parsed) for field in GROUNDING_FIELDS},
            }
        )
    return summary


def group_by(rows: list[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return dict(grouped)


OVERVIEW_COLUMNS = [
    ("n", "n"),
    ("questions", "questions"),
    ("answer_acc", "answer acc"),
    ("exact_match", "evidence EM"),
    ("minimal_set_success", "minimal-set"),
    ("complete", "completeness"),
    ("precision", "layer P"),
    ("recall", "layer R"),
    ("f1", "layer F1"),
    ("distractor_captured", "distractor"),
    ("excess", "excess"),
    ("invalid_citation", "invalid layers"),
    ("joint_acc", "joint acc"),
    ("format_fail", "format fail"),
]


def report(rows: list[dict[str, Any]], by_model: bool) -> str:
    out: list[str] = ["# Experiment A", ""]

    keys = [column for column, _ in OVERVIEW_COLUMNS]
    headers = ["condition"] + (["model"] if by_model else []) + [
        label for _, label in OVERVIEW_COLUMNS
    ]
    by_condition = group_by(rows, "condition")
    table_rows = []
    for condition in sorted(by_condition):
        condition_rows = by_condition[condition]
        groups = (
            sorted(group_by(condition_rows, "model").items())
            if by_model
            else [(None, condition_rows)]
        )
        for model, group in groups:
            summary = summarize(group)
            prefix = [condition] + ([model] if by_model else [])
            table_rows.append(prefix + [summary.get(key) for key in keys])
    out += [render_table(headers, table_rows), ""]

    for key, title in (("category", "By category"), ("demand_level", "By demand level")):
        if not any(row[key] is not None for row in rows):
            continue
        out += [f"## {title}", ""]
        headers = [key, "condition", "n", "answer acc", "evidence EM", "joint acc"]
        table_rows = []
        grouped = group_by(rows, key)
        for value in sorted(grouped, key=lambda v: (v is None, v)):
            for condition, group in sorted(group_by(grouped[value], "condition").items()):
                summary = summarize(group)
                table_rows.append(
                    [
                        value,
                        condition,
                        summary["n"],
                        summary["answer_acc"],
                        summary.get("exact_match"),
                        summary.get("joint_acc"),
                    ]
                )
        out += [render_table(headers, table_rows), ""]

    out += confidence_section(rows)
    return "\n".join(out)


def confidence_section(rows: list[dict[str, Any]]) -> list[str]:
    """Calibration and selective accuracy, when the model gave a confidence."""
    scored = [row for row in rows if row.get("confidence") is not None]
    if not scored:
        return []

    out = ["## Confidence", ""]
    buckets = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    table_rows = []
    for low, high in buckets:
        group = [row for row in scored if low <= row["confidence"] < high]
        if not group:
            continue
        summary = summarize(group)
        label = f"[{low}, 1.0]" if high > 1.0 else f"[{low}, {high})"
        table_rows.append(
            [label, summary["n"], summary["answer_acc"], summary.get("exact_match")]
        )
    out += [render_table(["confidence", "n", "answer acc", "evidence EM"], table_rows), ""]

    table_rows = []
    for threshold in (0.0, 0.5, 0.7, 0.9):
        group = [row for row in scored if row["confidence"] >= threshold]
        if not group:
            continue
        summary = summarize(group)
        table_rows.append(
            [
                f">= {threshold}",
                len(group) / len(scored),
                summary["answer_acc"],
                summary.get("joint_acc"),
            ]
        )
    out += [
        "Selective accuracy — answers kept at or above each confidence:",
        "",
        render_table(["threshold", "coverage", "answer acc", "joint acc"], table_rows),
        "",
    ]
    return out


def write_csv(rows: list[dict[str, Any]], path: str) -> None:
    fieldnames = list(rows[0]) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} conversation rows -> {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--log_folder", default="logs")
    parser.add_argument("--task", default="layered_vqa")
    parser.add_argument("--conv_type", default="full", help="Log folder the single-turn runs landed in")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="Dataset files, one per condition")
    parser.add_argument("--dataset_dir", default="../../data/derived/lic", help="Where those dataset files live, for category/demand metadata")
    parser.add_argument("--models", nargs="+", default=None, help="Only score these assistant models")
    parser.add_argument("--by_model", action="store_true", help="One row per condition and model instead of pooling models")
    parser.add_argument("--csv", default=None, help="Also write one row per conversation here")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = collect_rows(args)
    if not rows:
        raise SystemExit("no conversations matched — check --datasets and --log_folder")
    print(report(rows, by_model=args.by_model))
    if args.csv:
        write_csv(rows, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
