#!/usr/bin/env python3
"""Aggregate post-hoc judging into the paper's behavioural analyses.

Reads `logs/Final-*/posthoc/<judge>/**/*.jsonl` (written by `posthoc_judge.py`)
and emits one markdown report plus a CSV per table. Every analysis is broken out
by all six question categories, not the two or three a domain-level breakdown
would use.

Seven analyses, in the order the report prints them:

1. `cross-judge`    accuracy for every (assistant, judge) pair, and the
                    self-preference residual: how much a model's self-graded
                    score exceeds what that judge's leniency and that
                    assistant's ability predict. The original runs are entirely
                    on this diagonal, so the residual is the correction factor
                    for every headline number in the paper.
2. `first-attempt`  accuracy against how early the model first proposed *any*
                    answer, in fifths of the conversation. Committing early is
                    the behaviour; being wrong is the consequence.
3. `verbosity`      accuracy against relative response length, in quintiles
                    computed within each (model, condition) so a terse model is
                    not simply ranked below a florid one.
4. `bloat`          mean length of the n-th answer attempt, against the
                    single-turn baselines, per condition family.
5. `citation`       of the responses at turn Y, what fraction name a layer that
                    was introduced at turn X. Recency and staleness of evidence
                    use, read off the diagonal band.
6. `grounding`      the paper's own contribution: correct answers split by
                    whether the gold layers had actually arrived yet.
7. `churn`          answer stability -- changed, abandoned, recovered.

Usage:
    python posthoc_analysis.py --runs logs/Final-0 logs/Final-1 logs/Final-2 \
        --out logs/posthoc_report.md --csv_dir logs/posthoc_csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

DEFAULT_DATA_DIR = Path("../../data/derived/lic")
PROGRESS_BINS = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
VERBOSITY_BINS = ["shortest", "short", "median", "long", "longest"]

# (conv_type, dataset_fn) -> the condition label used throughout the paper.
CONDITIONS = {
    ("full", "layered_vqa_text.json"): "FULL",
    ("concat", "layered_vqa_text.json"): "CONCAT_Q",
    ("sharded", "layered_vqa_text.json"): "SHARDED_Q",
    ("recap-concat", "layered_vqa_text.json"): "RECAP_Q",
    ("snowball", "layered_vqa_text.json"): "SNOWBALL_Q",
    ("full", "layered_vqa_visual_canonical.json"): "CONCAT_V",
    ("sharded-scripted", "layered_vqa_visual_canonical.json"): "SHARDED_V",
    ("recap-concat", "layered_vqa_visual_canonical.json"): "RECAP_V",
    ("snowball-scripted", "layered_vqa_visual_canonical.json"): "SNOWBALL_V",
    ("sharded-scripted", "layered_vqa_aligned_vq.json"): "ALIGNED",
    ("sharded-scripted", "layered_vqa_misaligned_vq.json"): "MISALIGNED",
    ("sharded-scripted", "layered_vqa_visual_evidence_first.json"): "EVIDENCE_FIRST",
    ("sharded-scripted", "layered_vqa_visual_evidence_last.json"): "EVIDENCE_LAST",
    ("shuffle-concat", "layered_vqa_text.json"): "SHUFFLED_CONCAT",
    ("full", "layered_vqa_grounding.json"): "A2_ALL_LAYERS",
    ("full", "layered_vqa_oracle_gold.json"): "A3_ORACLE",
}
MULTITURN = {"SHARDED_Q", "RECAP_Q", "SNOWBALL_Q", "SHARDED_V", "RECAP_V",
             "SNOWBALL_V", "ALIGNED", "MISALIGNED", "EVIDENCE_FIRST", "EVIDENCE_LAST"}
SINGLE_TURN = {"FULL", "CONCAT_Q", "CONCAT_V", "SHUFFLED_CONCAT"}
LAYER_CONDITIONS = {"SHARDED_V", "ALIGNED", "MISALIGNED", "EVIDENCE_FIRST", "EVIDENCE_LAST"}


def condition_of(rec: dict) -> str:
    key = (rec.get("conv_type"), rec.get("dataset_fn"))
    return CONDITIONS.get(key, f"{rec.get('conv_type')}/{rec.get('dataset_fn')}")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def question_meta(data_dir: Path) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    for path in sorted(data_dir.glob("layered_vqa_*.json")):
        if path.name.endswith(".manifest.json"):
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            meta.setdefault(row["task_id"], {
                "category": row["meta"]["category"],
                "demand": row["meta"]["demand_level"],
            })
    return meta


def load_records(runs: Iterable[Path], judges: list[str] | None,
                 subdir: str = "posthoc") -> list[dict]:
    out = []
    for run in runs:
        root = run / subdir
        if not root.is_dir():
            continue
        for judge_dir in sorted(root.iterdir()):
            if not judge_dir.is_dir():
                continue
            if judges and judge_dir.name not in judges:
                continue
            for path in judge_dir.rglob("*.jsonl"):
                if path.name in ("judge_cache.jsonl", "errors.jsonl"):
                    continue
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("grader") == "error":
                        continue
                    rec["run"] = run.name
                    rec["condition"] = condition_of(rec)
                    out.append(rec)
    return out


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def rate(values: list[bool]) -> float:
    return sum(1 for v in values if v) / len(values) if values else float("nan")


def fmt(x: float, nd: int = 3) -> str:
    return "--" if x != x else f"{x:.{nd}f}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    rule = "|" + "|".join("---" for _ in headers) + "|"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return f"{line}\n{rule}\n{body}\n"


def quintile(index: int, total: int) -> str:
    """Which fifth of a sorted list position `index` falls in."""
    if total <= 1:
        return VERBOSITY_BINS[2]
    slot = min(4, int(5 * index / total))
    return VERBOSITY_BINS[slot]


class Csv:
    def __init__(self, directory: Path | None):
        self.dir = directory
        if directory:
            directory.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, headers: list[str], rows: list[list[Any]]) -> None:
        if not self.dir:
            return
        with (self.dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(headers)
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# 1. cross-judge matrix and self-preference
# ---------------------------------------------------------------------------

def cross_judge(records: list[dict], out: list[str], csvs: Csv) -> None:
    cell: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in records:
        cell[(r["assistant_model"], r["judge_model"])].append(r["is_correct"])

    assistants = sorted({a for a, _ in cell})
    judges = sorted({j for _, j in cell})
    if not assistants or not judges:
        return

    out.append("## 1. Cross-judge matrix\n")
    out.append(
        "Rows are the model that answered, columns the model that graded. The "
        "original runs occupy only the diagonal, so everything off it is new "
        "information about what those numbers were measuring.\n"
    )
    rows = []
    for a in assistants:
        cells = [fmt(rate(cell.get((a, j), []))) for j in judges]
        own = rate(cell.get((a, a), []))
        others = [rate(cell[(a, j)]) for j in judges if j != a and cell.get((a, j))]
        gap = own - statistics.mean(others) if others and own == own else float("nan")
        rows.append([a] + cells + [fmt(own), fmt(gap)])
    out.append(table(["assistant"] + judges + ["self", "self - others"], rows))
    csvs.write("cross_judge", ["assistant"] + judges + ["self", "self_minus_others"], rows)

    # Additive model: verdict ~ assistant ability + judge leniency. What the
    # diagonal exceeds that by is the part neither explains, which is the only
    # defensible reading of "self-preference" from a 6x6 grid.
    grand = rate([v for vals in cell.values() for v in vals])
    ability = {a: rate([v for (aa, _), vals in cell.items() if aa == a for v in vals])
               for a in assistants}
    leniency = {j: rate([v for (_, jj), vals in cell.items() if jj == j for v in vals])
                for j in judges}
    out.append("\n**Self-preference residual.** Observed self-graded accuracy minus "
               "`ability + leniency - grand mean`, the additive prediction for that "
               "cell. A positive residual is score a model gives itself that its own "
               "ability and its own leniency as a grader do not account for.\n")
    resid_rows = []
    for a in assistants:
        if (a, a) not in cell:
            continue
        observed = rate(cell[(a, a)])
        predicted = ability[a] + leniency.get(a, grand) - grand
        resid_rows.append([a, fmt(observed), fmt(predicted), fmt(observed - predicted),
                           str(len(cell[(a, a)]))])
    out.append(table(["model", "self-graded", "predicted", "residual", "n"], resid_rows))
    csvs.write("self_preference", ["model", "self_graded", "predicted", "residual", "n"], resid_rows)


# ---------------------------------------------------------------------------
# 2. premature answer attempts
# ---------------------------------------------------------------------------

def first_attempt(records: list[dict], meta: dict, out: list[str], csvs: Csv) -> None:
    multi = [r for r in records
             if r["condition"] in MULTITURN and r.get("first_attempt_bin")]
    if not multi:
        return

    out.append("\n## 2. Premature answer attempts\n")
    out.append(
        "Accuracy against how early the model first proposed *any* answer, right "
        "or wrong, as a fraction of the conversation. Multi-turn conditions only: "
        "a single-turn condition has one chance and cannot commit early. "
        "The pattern to read is monotonicity -- a model that answers in the first "
        "fifth is answering before the question is fully specified.\n"
    )

    by_model: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    by_cat: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    share: dict[str, list[str]] = defaultdict(list)
    for r in multi:
        b = r["first_attempt_bin"]
        by_model[r["assistant_model"]][b].append(r["is_correct"])
        cat = meta.get(r["task_id"], {}).get("category", "?")
        by_cat[cat][b].append(r["is_correct"])
        share[r["assistant_model"]].append(b)

    rows = []
    for model in sorted(by_model):
        rows.append([model] + [fmt(rate(by_model[model].get(b, []))) for b in PROGRESS_BINS])
    avg = [fmt(rate([v for m in by_model.values() for v in m.get(b, [])])) for b in PROGRESS_BINS]
    rows.append(["**average**"] + [f"**{x}**" for x in avg])
    out.append("**Accuracy by position of first answer attempt**\n")
    out.append(table(["model"] + PROGRESS_BINS, rows))
    csvs.write("first_attempt_by_model", ["model"] + PROGRESS_BINS, rows)

    cat_rows = []
    for cat in sorted(by_cat):
        cat_rows.append([cat] + [fmt(rate(by_cat[cat].get(b, []))) for b in PROGRESS_BINS])
    out.append("\n**By question category** — all six, not a domain subset.\n")
    out.append(table(["category"] + PROGRESS_BINS, cat_rows))
    csvs.write("first_attempt_by_category", ["category"] + PROGRESS_BINS, cat_rows)

    dist_rows = []
    for model in sorted(share):
        total = len(share[model])
        dist_rows.append([model] + [fmt(share[model].count(b) / total) for b in PROGRESS_BINS])
    out.append("\n**How often each model commits in each fifth** — the exposure "
               "behind the row above. A model whose mass sits in `0-20%` is "
               "answering first and listening afterwards.\n")
    out.append(table(["model"] + PROGRESS_BINS, dist_rows))
    csvs.write("first_attempt_distribution", ["model"] + PROGRESS_BINS, dist_rows)


# ---------------------------------------------------------------------------
# 3. verbosity
# ---------------------------------------------------------------------------

def verbosity(records: list[dict], meta: dict, out: list[str], csvs: Csv) -> None:
    # Quintiles are computed within each (model, condition): "relative" means
    # relative to that model's own habit in that setting, so a tersely-tuned
    # model does not simply occupy the short bins everywhere.
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        if r.get("total_chars"):
            groups[(r["assistant_model"], r["condition"])].append(r)
    if not groups:
        return

    tagged: list[tuple[dict, str]] = []
    for rows in groups.values():
        rows.sort(key=lambda r: r["total_chars"])
        for i, r in enumerate(rows):
            tagged.append((r, quintile(i, len(rows))))

    out.append("\n## 3. Relative assistant verbosity\n")
    out.append(
        "Every conversation carries a verbosity tag, assigned by ranking it "
        "against the same model's other conversations *in the same condition* "
        "and cutting into fifths. Reading it per (model, condition) rather than "
        "globally is what keeps the tag about verbosity instead of about which "
        "model is chattier or which condition is longer.\n"
    )

    by_cat: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    by_cond: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r, tag in tagged:
        cat = meta.get(r["task_id"], {}).get("category", "?")
        by_cat[cat][tag].append(r["is_correct"])
        by_cond[r["condition"]][tag].append(r["is_correct"])

    rows = [[c] + [fmt(rate(by_cat[c].get(t, []))) for t in VERBOSITY_BINS] for c in sorted(by_cat)]
    avg = [fmt(rate([v for c in by_cat.values() for v in c.get(t, [])])) for t in VERBOSITY_BINS]
    rows.append(["**average**"] + [f"**{x}**" for x in avg])
    out.append("**By question category**\n")
    out.append(table(["category"] + VERBOSITY_BINS, rows))
    csvs.write("verbosity_by_category", ["category"] + VERBOSITY_BINS, rows)

    cond_rows = [[c] + [fmt(rate(by_cond[c].get(t, []))) for t in VERBOSITY_BINS]
                 for c in sorted(by_cond)]
    out.append("\n**By condition**\n")
    out.append(table(["condition"] + VERBOSITY_BINS, cond_rows))
    csvs.write("verbosity_by_condition", ["condition"] + VERBOSITY_BINS, cond_rows)


# ---------------------------------------------------------------------------
# 4. answer bloat
# ---------------------------------------------------------------------------

def bloat(records: list[dict], out: list[str], csvs: Csv) -> None:
    by_index: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    baseline: dict[str, list[int]] = defaultdict(list)
    for r in records:
        cond = r["condition"]
        if cond in SINGLE_TURN:
            baseline[cond].extend(x["chars"] for x in r.get("attempt_lengths", []))
            continue
        for x in r.get("attempt_lengths", []):
            by_index[cond][x["attempt_index"]].append(x["chars"])
    if not by_index:
        return

    out.append("\n## 4. Answer bloat across attempts\n")
    out.append(
        "Mean characters in the n-th answer attempt. The single-turn conditions "
        "have exactly one attempt and appear as the baseline row: if multi-turn "
        "attempts start near it and climb, the model is padding as the "
        "conversation drags rather than answering a harder question.\n"
    )
    max_index = min(8, max((i for d in by_index.values() for i in d), default=0))
    headers = ["condition", "n"] + [f"#{i}" for i in range(1, max_index + 1)]
    rows = []
    for cond in sorted(by_index):
        counts = by_index[cond]
        total = sum(len(v) for v in counts.values())
        cells = []
        for i in range(1, max_index + 1):
            vals = counts.get(i, [])
            cells.append(f"{statistics.mean(vals):.0f}" if vals else "--")
        rows.append([cond, str(total)] + cells)
    for cond in sorted(baseline):
        vals = baseline[cond]
        rows.append([f"{cond} (single turn)", str(len(vals)),
                     f"{statistics.mean(vals):.0f}"] + ["--"] * (max_index - 1))
    out.append(table(headers, rows))
    csvs.write("answer_bloat", headers, rows)


# ---------------------------------------------------------------------------
# 5. citation recency
# ---------------------------------------------------------------------------

def citation(records: list[dict], out: list[str], csvs: Csv) -> None:
    # cell[(y, x)] = how many turn-y responses named a layer introduced at turn x
    hits: dict[tuple[int, int], int] = defaultdict(int)
    turns_seen: dict[int, int] = defaultdict(int)
    for r in records:
        if r["condition"] not in LAYER_CONDITIONS:
            continue
        intro = {int(k): v for k, v in (r.get("layer_intro_turn") or {}).items()}
        if not intro:
            continue
        for y, mentioned in enumerate(r.get("layer_mentions") or [], start=1):
            turns_seen[y] += 1
            for x in {intro.get(lid) for lid in mentioned}:
                if x:
                    hits[(y, x)] += 1
    if not turns_seen:
        return

    out.append("\n## 5. Which turn's evidence a response actually uses\n")
    out.append(
        "Row *y* is a response at assistant turn *y*; column *x* is a layer that "
        "arrived at turn *x*. The cell is the share of turn-*y* responses that "
        "name a layer introduced at turn *x*, matched on the layer's annotated "
        "label. Rows do not sum to 1: a response may name several layers or "
        "none, and a label the annotation never assigned cannot be matched at "
        "all -- the background layer carries no label, and turn 1 is the "
        "question turn, which delivers no layer at all, so those columns are "
        "structurally empty rather than unused. Read the band, not the cell -- "
        "mass on the diagonal is a model "
        "that talks about whatever just arrived, mass in column 1 is a model "
        "still anchored on its first look at the scene.\n"
    )
    max_turn = min(8, max(turns_seen))
    headers = ["turn y \\ layer from x", "n"] + [str(x) for x in range(1, max_turn + 1)]
    rows = []
    for y in range(1, max_turn + 1):
        n = turns_seen.get(y, 0)
        cells = [fmt(hits.get((y, x), 0) / n, 2) if n else "--" for x in range(1, max_turn + 1)]
        rows.append([str(y), str(n)] + cells)
    out.append(table(headers, rows))
    csvs.write("citation_recency", headers, rows)


# ---------------------------------------------------------------------------
# 6. evidence grounding  (the contribution)
# ---------------------------------------------------------------------------

def grounding(records: list[dict], meta: dict, out: list[str], csvs: Csv) -> None:
    usable = [r for r in records if r.get("has_layer_schedule")]
    if not usable:
        return

    out.append("\n## 6. Grounded vs ungrounded correct answers\n")
    out.append(
        "A correct answer is *ungrounded* when it was given at a turn where at "
        "least one gold layer had not yet been delivered. Standard accuracy "
        "cannot separate the two: it credits a model equally for reading the "
        "evidence it was handed and for producing the answer before that "
        "evidence arrived. `ungrounded share` is the fraction of a condition's "
        "correct answers that the evidence did not yet support, and it is the "
        "amount by which that condition's headline accuracy overstates "
        "evidence-following.\n"
    )

    def block(key_fn, label: str, name: str) -> None:
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in usable:
            groups[key_fn(r)].append(r)
        rows = []
        for key in sorted(groups):
            g = groups[key]
            correct = [r for r in g if r["is_correct"]]
            ungrounded = [r for r in correct if r["ungrounded_correct"]]
            rows.append([
                key, str(len(g)),
                fmt(len(correct) / len(g)),
                fmt(len(ungrounded) / len(correct)) if correct else "--",
                fmt((len(correct) - len(ungrounded)) / len(g)),
            ])
        headers = [label, "n", "accuracy", "ungrounded share", "grounded accuracy"]
        out.append(f"\n**By {label}**\n")
        out.append(table(headers, rows))
        csvs.write(name, headers, rows)

    block(lambda r: r["condition"], "condition", "grounding_by_condition")
    block(lambda r: r["assistant_model"], "model", "grounding_by_model")
    block(lambda r: meta.get(r["task_id"], {}).get("category", "?"),
          "category", "grounding_by_category")

    aligned = [r for r in usable if r["condition"] == "ALIGNED"]
    misaligned = [r for r in usable if r["condition"] == "MISALIGNED"]
    if aligned and misaligned:
        out.append(
            "\n**Alignment interaction.** The misaligned arm delivers the same "
            "layers on turns that do not support the shard being asked. If its "
            "correct answers are more often ungrounded, then part of what looks "
            "like a modest alignment cost is really a shift in *how* the "
            "remaining correct answers were reached.\n"
        )
        rows = []
        for label, group in (("ALIGNED", aligned), ("MISALIGNED", misaligned)):
            correct = [r for r in group if r["is_correct"]]
            ung = [r for r in correct if r["ungrounded_correct"]]
            rows.append([label, str(len(group)), fmt(len(correct) / len(group)),
                         fmt(len(ung) / len(correct)) if correct else "--"])
        out.append(table(["arm", "n", "accuracy", "ungrounded share"], rows))
        csvs.write("grounding_alignment", ["arm", "n", "accuracy", "ungrounded_share"], rows)


# ---------------------------------------------------------------------------
# 7. answer churn
# ---------------------------------------------------------------------------

def churn(records: list[dict], out: list[str], csvs: Csv) -> None:
    multi = [r for r in records if r["condition"] in MULTITURN and r.get("n_attempts")]
    if not multi:
        return

    out.append("\n## 7. Answer stability\n")
    out.append(
        "What happens to an answer once it exists. `changed` is any conversation "
        "that proposed more than one distinct answer; `abandoned` had the right "
        "answer at some turn and a different one at the end; `recovered` was "
        "wrong at its first attempt and right at the end. A high abandoned rate "
        "next to a high accuracy means the benchmark is scoring answers the "
        "model itself no longer stood behind by the final turn.\n"
    )
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in multi:
        groups[r["condition"]].append(r)
    headers = ["condition", "n", "mean attempts", "changed", "abandoned", "recovered"]
    rows = []
    for cond in sorted(groups):
        g = groups[cond]
        recovered = [r for r in g if not r["first_attempt_correct"] and r["final_turn_correct"]]
        rows.append([
            cond, str(len(g)),
            f"{statistics.mean([r['n_attempts'] for r in g]):.2f}",
            fmt(rate([r["changed_answer"] for r in g])),
            fmt(rate([r["abandoned"] for r in g])),
            fmt(len(recovered) / len(g)),
        ])
    out.append(table(headers, rows))
    csvs.write("answer_churn", headers, rows)


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", nargs="+", default=["logs/Final-0", "logs/Final-1", "logs/Final-2"])
    parser.add_argument("--judges", nargs="*", default=None,
                        help="Only these judges (default: every judge found)")
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--subdir", default="posthoc",
                        help="Which directory under each run holds the graded "
                             "records; must match what posthoc_judge.py wrote")
    parser.add_argument("--out", default="logs/posthoc_report.md")
    parser.add_argument("--csv_dir", default="logs/posthoc_csv")
    args = parser.parse_args()

    runs = [Path(r) for r in args.runs]
    records = load_records(runs, args.judges, args.subdir)
    if not records:
        print(f"No graded records under <run>/{args.subdir}/. Run posthoc_judge.py with the same --subdir first.")
        return 1
    meta = question_meta(Path(args.data_dir))
    csvs = Csv(Path(args.csv_dir) if args.csv_dir else None)

    judges = sorted({r["judge_model"] for r in records})
    assistants = sorted({r["assistant_model"] for r in records})
    out: list[str] = [
        "# Post-hoc behavioural analysis\n",
        f"{len(records):,} graded conversations | "
        f"{len(assistants)} assistants x {len(judges)} judges | "
        f"runs: {', '.join(r.name for r in runs)}\n",
    ]

    cross_judge(records, out, csvs)
    first_attempt(records, meta, out, csvs)
    verbosity(records, meta, out, csvs)
    bloat(records, out, csvs)
    citation(records, out, csvs)
    grounding(records, meta, out, csvs)
    churn(records, out, csvs)

    text = "\n".join(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    if args.csv_dir:
        print(f"wrote CSVs to {args.csv_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
