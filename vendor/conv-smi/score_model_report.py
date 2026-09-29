"""Write one markdown report per model covering Experiment A *and* Experiment B.

`run_all_open_models.sh` calls this right after a model finishes, before its
weights are unloaded, so a long multi-model run leaves a readable per-model
result file behind at every step instead of only at the end.

The two halves are the same reports `score_experiment_a.py` and
`score_experiment_b.py` print, rendered into one file so a model's answer
accuracy, its grounding metrics and its delivery-condition contrasts are read
together. Nothing is recomputed here: both scorers read the same logs, and a
condition that has not run yet is reported as missing rather than skipped
silently.

    python score_model_report.py --log_folder logs/Layered-VQA-0 \
        --models qwen3.5-9b
    python score_model_report.py --log_folder logs/Layered-VQA-0 \
        --models qwen3.5-9b --out logs/Layered-VQA-0/results_qwen3.5-9b.md
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

import score_experiment_a
import score_experiment_b


def experiment_a_section(args: argparse.Namespace, model: str) -> str:
    a_args = score_experiment_a.parse_args(
        [
            "--log_folder", args.log_folder,
            "--task", args.task,
            "--conv_type", args.conv_type,
            "--dataset_dir", args.dataset_dir,
            "--models", model,
            "--by_model",
        ]
    )
    try:
        rows = score_experiment_a.collect_rows(a_args)
    except SystemExit as exc:  # no `full` log folder yet
        return f"# Experiment A\n\n*Not available: {exc}*\n"
    if not rows:
        return "# Experiment A\n\n*No conversations logged for this model yet.*\n"
    return score_experiment_a.report(rows, by_model=True)


def experiment_b_section(args: argparse.Namespace, model: str) -> str:
    b_args = score_experiment_b.parse_args(
        [
            "--log_folder", args.log_folder,
            "--task", args.task,
            "--dataset_dir", args.dataset_dir,
            "--models", model,
            "--by_model",
        ]
    )
    text, _ = score_experiment_b.build_report(b_args)
    return text


def build(args: argparse.Namespace, model: str) -> str:
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = [
        f"# {model}",
        "",
        f"Layered-VQA results for `{model}`, generated {stamp} from "
        f"`{args.log_folder}`.",
        "",
        "Both experiments are scored from the same conversation logs: "
        "Experiment A measures what the model can do when the evidence is "
        "handed to it, Experiment B measures what delivering that evidence "
        "across turns costs.",
        "",
        "---",
        "",
    ]
    return "\n".join(
        header
        + [experiment_a_section(args, model), "", "---", "", experiment_b_section(args, model)]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--log_folder", default="logs")
    parser.add_argument("--task", default="layered_vqa")
    parser.add_argument("--conv_type", default="full", help="Log folder the Experiment-A single-turn runs landed in")
    parser.add_argument("--dataset_dir", default="../../data/derived/lic")
    parser.add_argument("--models", nargs="+", required=True, help="Write one report per model")
    parser.add_argument(
        "--out",
        default=None,
        help="Output path. Only valid with a single --models value; otherwise "
        "each report goes to <log_folder>/results_<model>.md",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.out and len(args.models) > 1:
        raise SystemExit("--out takes a single --models value")

    for model in args.models:
        path = Path(args.out) if args.out else Path(args.log_folder) / f"results_{model}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(build(args, model), encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
