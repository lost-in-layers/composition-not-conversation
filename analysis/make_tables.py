#!/usr/bin/env python3
"""Regenerate every LaTeX table in tables.tex from the Final-* simulation logs.

Run from anywhere:  python overleaf/make_tables.py

Design notes that matter for reading the output:

* **Missing conversations count as wrong.** A prompt that overflowed the served
  context window raises before a verdict is logged, so the sample disappears
  from the JSONL rather than scoring zero. Silently dropping it would reward the
  model for the conversations it could not hold. Every condition is therefore
  scored over the full 300-question dataset, with unlogged questions counted
  incorrect. Table `tab:coverage` reports exactly how many that is per cell, so
  the assumption stays visible and reversible. This is a v0-draft convention:
  the right fix is re-running `internvl3.5-8b` at a longer context.
* **Runs are replicates, not folds.** Final-0/1/2 use identical settings and
  differ only in sampling. Point estimates pool all three; the reported spread
  is the SD across the three run-level accuracies.
* **Contrasts are paired per question.** Both arms of every contrast saw the
  same 300 questions, so the tests pair on `task_id` rather than comparing
  independent means. The bootstrap resamples questions (carrying all three runs
  of a question together) so the resampling unit stays the question.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOGS = REPO / "vendor" / "conv-smi" / "logs"
DATA = REPO / "data" / "derived" / "lic"
OUT = Path(__file__).resolve().parent / "tables.tex"
# Same numbers, same captions, no condition glyphs -- for a paper whose preamble
# has no TikZ, or if the glyphs turn out to crowd the final column widths.
OUT_PLAIN = Path(__file__).resolve().parent / "tables_plain.tex"

RUNS = ["Final-0", "Final-1", "Final-2"]
N_BOOT = 4000

# Display order = descending overall accuracy, set after the numbers are in.
MODELS = [
    ("pixtral-12b", r"Pixtral-12B"),
    ("qwen3.6-27b", r"Qwen3.6-27B"),
    ("qwen2.5-vl-7b", r"Qwen2.5-VL-7B"),
    ("qwen3.5-9b", r"Qwen3.5-9B"),
    ("internvl3.5-8b", r"InternVL3.5-8B"),
    ("ministral3-8b-instruct", r"Ministral-3-8B"),
]
MODEL_IDS = [m for m, _ in MODELS]

# Which glyph labels which condition. Kept beside the condition tables rather
# than inside them so a renamed condition cannot silently keep an icon that no
# longer depicts what it does.
ICON_A = {"A1": r"\icRGB", "A2": r"\icAll", "A3": r"\icOracle"}
ICON_B = {
    "B1": r"\icFull", "B2": r"\icConcat", "B3": r"\icSharded",
    "B4": r"\icRecap", "B5": r"\icSnowball",
    "B6": r"\icConcat", "B7": r"\icSharded", "B8": r"\icRecap",
    "B9": r"\icSnowball",
    "B10": r"\icAligned", "B11": r"\icMisaligned",
    "B12": r"\icEvFirst", "B13": r"\icEvLast", "B14": r"\icShuffled",
}
ICON_FAMILY = {"Text": r"\icText", "Visual": r"\icVisual",
               "Layered": r"\icLayered", "Order": r"\icOrder"}

# Flipped by main(); the table builders read it at call time, so one process
# emits both variants. Without glyphs the conditions fall back to their B-numbers,
# which is why those keys stay in EXP_B rather than being replaced by icons.
ICONS = True


def ia(key: str) -> str:
    """Experiment-A column glyph, or nothing."""
    return ICON_A[key] if ICONS else ""


def ib(key: str) -> str:
    """Experiment-B row label: the glyph, or the plain B-number."""
    return ICON_B[key] if ICONS else key


def ifam(family: str) -> str:
    return ICON_FAMILY[family] + r"\," if ICONS else ""


def glyph_row_a() -> str:
    """The icon strip above the Experiment-A column heads, or nothing."""
    if not ICONS:
        return ""
    return (f'& {ICON_A["A1"]} & {ICON_A["A2"]} & {ICON_A["A3"]}\n'
            f'      & {ICON_A["A2"]} & {ICON_A["A3"]} \\\\[1pt]\n')


LEGEND_A = r"""
Column glyphs show what is on the table: a framed scene (\icRGB), the full layer
stack (\icAll), and the same stack with only the gold subset actually present
(\icOracle)."""

LEGEND_B = r"""
\textbf{Reading the glyphs:} a rounded bar is one conversation turn and the
light-to-dark ramp is shard order, so bars fused into one block mean a single
turn carrying every shard (\icConcat) while separated bars mean one shard per
turn (\icSharded); a trailing heavy bar is the restatement turn (\icRecap) and
widening bars are accumulation (\icSnowball). A square above a bar is a layer
image attached to that turn --- seated on the bar when it supports that shard
(\icAligned), lifted and crossed when it does not (\icMisaligned)."""


def legend(text: str) -> str:
    return text if ICONS else ""

EXP_A = [
    ("A1", r"\textsc{rgb}", "full", "layered_vqa_text.json"),
    ("A2", r"\textsc{all-layers}", "full", "layered_vqa_grounding.json"),
    ("A3", r"\textsc{oracle}", "full", "layered_vqa_oracle_gold.json"),
]

# label, conv_type, dataset_fn, family, turns-are-interesting
EXP_B = [
    ("B1", r"\textsc{full}", "full", "layered_vqa_text.json", "Text"),
    ("B2", r"\textsc{concat}", "concat", "layered_vqa_text.json", "Text"),
    ("B3", r"\textsc{sharded}", "sharded", "layered_vqa_text.json", "Text"),
    ("B4", r"\textsc{recap}", "recap-concat", "layered_vqa_text.json", "Text"),
    ("B5", r"\textsc{snowball}", "snowball", "layered_vqa_text.json", "Text"),
    ("B6", r"\textsc{concat}", "full", "layered_vqa_visual_canonical.json", "Visual"),
    ("B7", r"\textsc{sharded}", "sharded-scripted", "layered_vqa_visual_canonical.json", "Visual"),
    ("B8", r"\textsc{recap}", "recap-concat", "layered_vqa_visual_canonical.json", "Visual"),
    ("B9", r"\textsc{snowball}", "snowball-scripted", "layered_vqa_visual_canonical.json", "Visual"),
    ("B10", r"\textsc{aligned}", "sharded-scripted", "layered_vqa_aligned_vq.json", "Layered"),
    ("B11", r"\textsc{misaligned}", "sharded-scripted", "layered_vqa_misaligned_vq.json", "Layered"),
    ("B12", r"\textsc{evidence-first}", "sharded-scripted", "layered_vqa_visual_evidence_first.json", "Order"),
    ("B13", r"\textsc{evidence-last}", "sharded-scripted", "layered_vqa_visual_evidence_last.json", "Order"),
    ("B14", r"\textsc{shuffled-concat}", "shuffle-concat", "layered_vqa_text.json", "Order"),
]

CONTRASTS = [
    ("Modality cost", r"\textsc{concat-v} $-$ \textsc{concat-q}", "B6", "B2",
     "Replacing the text question with layer renders"),
    ("Recap recovery (V)", r"\textsc{recap-v} $-$ \textsc{sharded-v}", "B8", "B7",
     "Re-sending the full RGB after a sharded visual conversation"),
    ("Sharding gain (V)", r"\textsc{sharded-v} $-$ \textsc{concat-v}", "B7", "B6",
     "Delivering layers one turn at a time vs.\\ all at once"),
    ("Alignment cost", r"\textsc{misaligned} $-$ \textsc{aligned}", "B11", "B10",
     "Pairing each shard with a layer it does not support"),
    ("Evidence order", r"\textsc{ev-last} $-$ \textsc{ev-first}", "B13", "B12",
     "Withholding evidence until after the question"),
    ("Sharding cost (Q)", r"\textsc{sharded-q} $-$ \textsc{concat-q}", "B3", "B2",
     "Splitting the text question across turns"),
    ("Recap recovery (Q)", r"\textsc{recap-q} $-$ \textsc{sharded-q}", "B4", "B3",
     "Restating the full text question at the end"),
    ("Shard order (Q)", r"\textsc{shuffled} $-$ \textsc{concat-q}", "B14", "B2",
     "Permuting the non-intent shards before concatenating"),
]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def dataset_meta() -> tuple[dict[str, set[str]], dict[str, dict]]:
    """task_ids per dataset file, and per-question metadata."""
    ids, meta = {}, {}
    for path in DATA.glob("layered_vqa_*.json"):
        if path.name.endswith(".manifest.json"):
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        ids[path.name] = {r["task_id"] for r in rows}
        for r in rows:
            meta.setdefault(r["task_id"], {
                "category": r["meta"]["category"],
                "demand": r["meta"]["demand_level"],
            })
    return ids, meta


def load() -> tuple[dict, dict, dict]:
    """verdicts[(run, model, key)] -> {task_id: bool};  plus turns and grounding."""
    # A list, not a dict: A1 and B1 are the same (conv_type, dataset) pair and
    # both must be registered, so keying by the pair would drop one of them.
    want = [(ct, fn, key) for key, _, ct, fn in EXP_A]
    want += [(ct, fn, key) for key, _, ct, fn, _ in EXP_B]

    verdicts = defaultdict(dict)
    turns = defaultdict(list)
    grounding = defaultdict(list)

    for run in RUNS:
        base = LOGS / run / "layered_vqa"
        if not base.is_dir():
            continue
        for path in base.rglob("*.jsonl"):
            conv_type = path.parent.name
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                model = rec.get("assistant_model")
                if model not in MODEL_IDS:
                    continue
                for ct, fn, key in want:
                    if ct != conv_type or rec.get("dataset_fn") != fn:
                        continue
                    verdicts[(run, model, key)][rec["task_id"]] = bool(rec["is_correct"])
                    n_asst = sum(1 for e in rec["trace"] if e.get("role") == "assistant")
                    turns[(model, key)].append(n_asst)
                    if key == "A2":
                        for e in rec["trace"]:
                            if e.get("role") != "log":
                                continue
                            c = e.get("content", {})
                            if c.get("type") != "answer-evaluation":
                                continue
                            ev = c.get("evaluation_return", {})
                            lm, ls = ev.get("layer_metrics"), ev.get("layer_selection")
                            if lm:
                                grounding[model].append({
                                    **lm,
                                    "joint": bool(ev.get("joint_correct")),
                                    "format_ok": bool((ls or {}).get("format_ok")),
                                })
    return verdicts, turns, grounding


DATASET_IDS, QMETA = dataset_meta()
VERDICTS, TURNS, GROUNDING = load()

KEY_FN = {k: fn for k, _, _, fn in EXP_A}
KEY_FN.update({k: fn for k, _, _, fn, _ in EXP_B})


def scored(run: str, model: str, key: str) -> dict[str, bool]:
    """Verdicts over the FULL dataset; questions with no logged run count wrong."""
    logged = VERDICTS.get((run, model, key), {})
    return {tid: logged.get(tid, False) for tid in DATASET_IDS[KEY_FN[key]]}


def acc(model: str, key: str) -> float:
    hits = tot = 0
    for run in RUNS:
        v = scored(run, model, key)
        hits += sum(v.values())
        tot += len(v)
    return hits / tot if tot else float("nan")


def acc_sd(model: str, key: str) -> float:
    per_run = []
    for run in RUNS:
        v = scored(run, model, key)
        per_run.append(sum(v.values()) / len(v))
    return statistics.stdev(per_run)


def missing(model: str, key: str) -> int:
    return sum(len(DATASET_IDS[KEY_FN[key]]) - len(VERDICTS.get((run, model, key), {}))
               for run in RUNS)


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def normal_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def mcnemar(model: str, cond: str, ref: str) -> tuple[int, int, float]:
    b = c = 0
    for run in RUNS:
        A, B = scored(run, model, cond), scored(run, model, ref)
        for tid in A.keys() & B.keys():
            if A[tid] and not B[tid]:
                b += 1
            elif B[tid] and not A[tid]:
                c += 1
    n = b + c
    if n == 0:
        return b, c, 1.0
    z = max((abs(b - c) - 1) / math.sqrt(n), 0.0)
    return b, c, 2 * (1 - normal_cdf(z))


def paired_rows(model: str, cond: str, ref: str) -> list[tuple[float, float]]:
    ids = sorted(DATASET_IDS[KEY_FN[cond]] & DATASET_IDS[KEY_FN[ref]])
    rows = []
    for tid in ids:
        a = sum(scored(r, model, cond)[tid] for r in RUNS) / len(RUNS)
        b = sum(scored(r, model, ref)[tid] for r in RUNS) / len(RUNS)
        rows.append((a, b))
    return rows


def bootstrap(rows, seed=0) -> tuple[float, float, float]:
    if not rows:
        return float("nan"), float("nan"), float("nan")
    rnd = random.Random(seed)
    n = len(rows)
    point = sum(a - b for a, b in rows) / n
    deltas = []
    for _ in range(N_BOOT):
        s = 0.0
        for _ in range(n):
            a, b = rows[rnd.randrange(n)]
            s += a - b
        deltas.append(s / n)
    deltas.sort()
    return point, deltas[int(0.025 * N_BOOT)], deltas[int(0.975 * N_BOOT)]


def pooled_contrast(cond: str, ref: str, seed=0):
    """Across all models: cluster-bootstrap by question, McNemar over all pairs."""
    per_q = defaultdict(list)
    b = c = 0
    signs = []
    for model in MODEL_IDS:
        ids = DATASET_IDS[KEY_FN[cond]] & DATASET_IDS[KEY_FN[ref]]
        d = 0
        for run in RUNS:
            A, B = scored(run, model, cond), scored(run, model, ref)
            for tid in ids:
                per_q[tid].append(A[tid] - B[tid])
                d += A[tid] - B[tid]
                if A[tid] and not B[tid]:
                    b += 1
                elif B[tid] and not A[tid]:
                    c += 1
        signs.append(d)
    qs = list(per_q)
    point = sum(sum(v) for v in per_q.values()) / sum(len(v) for v in per_q.values())
    rnd = random.Random(seed)
    deltas = []
    for _ in range(N_BOOT):
        num = den = 0
        for _ in range(len(qs)):
            v = per_q[qs[rnd.randrange(len(qs))]]
            num += sum(v)
            den += len(v)
        deltas.append(num / den)
    deltas.sort()
    n = b + c
    p = 1.0 if n == 0 else 2 * (1 - normal_cdf(max((abs(b - c) - 1) / math.sqrt(n), 0.0)))
    agree = sum(1 for s in signs if s < 0) if point < 0 else sum(1 for s in signs if s > 0)
    return point, deltas[int(0.025 * N_BOOT)], deltas[int(0.975 * N_BOOT)], p, agree


def stars(p: float) -> str:
    return r"$^{\ast\ast\ast}$" if p < 0.001 else (
        r"$^{\ast\ast}$" if p < 0.01 else (r"$^{\ast}$" if p < 0.05 else ""))


def fmt(x: float, nd=3) -> str:
    return "--" if x != x else f"{x:.{nd}f}".lstrip("0") if 0 < abs(x) < 1 else f"{x:.{nd}f}"


def signed(x: float, nd=3) -> str:
    if x != x:
        return "--"
    s = f"{abs(x):.{nd}f}"
    s = s.lstrip("0") if abs(x) < 1 else s
    return ("$-$" if x < 0 else "$+$") + s


def heat(x: float, lo: float, hi: float) -> str:
    """Cell shading, strongest at hi. Kept light so print stays legible."""
    if x != x:
        return ""
    t = 0.0 if hi == lo else max(0.0, min(1.0, (x - lo) / (hi - lo)))
    return f"\\cellcolor{{tabhi!{int(round(t * 55))}}}"


# --------------------------------------------------------------------------
# table builders
# --------------------------------------------------------------------------

def table_exp_a() -> str:
    lo = min(acc(m, k) for m in MODEL_IDS for k, *_ in EXP_A)
    hi = max(acc(m, k) for m in MODEL_IDS for k, *_ in EXP_A)
    L = []
    for model, disp in MODELS:
        cells = []
        for key, *_ in EXP_A:
            a = acc(model, key)
            cells.append(f"{heat(a, lo, hi)}{fmt(a)}\\,\\tiny{{$\\pm${acc_sd(model,key):.3f}}}")
        _, _, p21 = mcnemar(model, "A2", "A1")
        _, _, p31 = mcnemar(model, "A3", "A1")
        d21 = acc(model, "A2") - acc(model, "A1")
        d31 = acc(model, "A3") - acc(model, "A1")
        L.append(f"{disp} & " + " & ".join(cells)
                 + f" & {signed(d21)}{stars(p21)} & {signed(d31)}{stars(p31)} \\\\")
    body = "\n".join(L)
    return rf"""
\begin{{table}}[t]
\centering
\caption{{\textbf{{Experiment A --- what the models can do when evidence is handed to them.}}
Answer accuracy over $300$ questions $\times$ $3$ replicate runs ($n=900$);
$\pm$ is the standard deviation across the three runs.
\textsc{{rgb}} shows the composed scene, \textsc{{all-layers}} shows every layer and asks
the model to name the ones it used, \textsc{{oracle}} shows only the gold minimal
sufficient layer set. The last column is the result that matters:
\emph{{giving a model exactly the evidence it needs is worse than giving it the whole
image}}, for every model, in every run. \textsc{{oracle}} is therefore not an upper bound,
and the visual conditions confound evidence delivery with render legibility.
Significance is McNemar over paired questions;
$^{{\ast}}p<.05$, $^{{\ast\ast}}p<.01$, $^{{\ast\ast\ast}}p<.001$.{legend(LEGEND_A)}}}
\label{{tab:exp-a}}
\small
\begin{{tabular}}{{l ccc cc}}
\toprule
& \multicolumn{{3}}{{c}}{{Answer accuracy}} & \multicolumn{{2}}{{c}}{{$\Delta$ vs.\ \textsc{{rgb}}}} \\
\cmidrule(lr){{2-4}} \cmidrule(lr){{5-6}}
{glyph_row_a()}Model & \textsc{{rgb}} & \textsc{{all-layers}} & \textsc{{oracle}}
      & \textsc{{all-layers}} & \textsc{{oracle}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def table_grounding() -> str:
    L = []
    for model, disp in MODELS:
        g = GROUNDING[model]
        n = len(g)
        if not n:
            continue
        def mean(f):
            return sum(f(x) for x in g) / n
        L.append(
            f"{disp} & {fmt(acc(model,'A2'))} & {fmt(mean(lambda x: x['precision']))} & "
            f"{fmt(mean(lambda x: x['recall']))} & {fmt(mean(lambda x: x['f1']))} & "
            f"{fmt(mean(lambda x: x['exact_match']))} & "
            f"{fmt(mean(lambda x: x['complete']))} & "
            f"{fmt(mean(lambda x: x['excess_rate']))} & "
            f"{fmt(mean(lambda x: x['distractor_captured']))} & "
            f"{fmt(mean(lambda x: bool(x['invalid_ids'])))} & "
            f"{fmt(mean(lambda x: x['joint']))} \\\\")
    body = "\n".join(L)
    return rf"""
\begin{{table}}[t]
\centering
\caption{{\textbf{{Evidence selection under all-layers delivery}} (Experiment A2, \textsc{{all-layers}}).
Every column except \emph{{Ans.}} is computed programmatically from the predicted layer
set against the annotation --- no judge is involved. \emph{{EM}} is exact evidence match,
\emph{{Comp.}} is recall of a complete sufficient set, \emph{{Exc.}} is the rate of layers
cited beyond the gold set, \emph{{Dist.}} is the rate of citing an annotated
distractor, \emph{{Inv.}} is citing a layer id that does not exist, and \emph{{Joint}}
requires a correct answer \emph{{and}} an exactly correct evidence set.
Minimal-sufficient-set success is omitted: on this annotation the first minimal
sufficient set equals the gold set for all $300$ questions, so it is identical to
\emph{{EM}} column-for-column. \textbf{{Read this against Table~\ref{{tab:exp-a}}:}}
Pixtral-12B leads every judge-scored condition there, yet leads \emph{{none}} of the
judge-free columns here --- fourth of six on F1, third on EM, second on
\emph{{Joint}}. Qwen3.6-27B leads both. A single-judge protocol in which each model grades itself cannot
distinguish that pattern from self-preference, which is why the cross-judge
study is the first item of future work.}}
\label{{tab:grounding}}
\small
\setlength{{\tabcolsep}}{{4.5pt}}
\begin{{tabular}}{{l c ccc cc ccc c}}
\toprule
& & \multicolumn{{3}}{{c}}{{Layer set quality}} & \multicolumn{{2}}{{c}}{{Sufficiency}}
& \multicolumn{{3}}{{c}}{{Error modes}} & \\
\cmidrule(lr){{3-5}} \cmidrule(lr){{6-7}} \cmidrule(lr){{8-10}}
Model & Ans. & P & R & F1 & EM & Comp. & Exc. & Dist. & Inv. & Joint \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def table_exp_b() -> str:
    all_acc = [acc(m, k) for m in MODEL_IDS for k, *_ in EXP_B]
    lo, hi = min(all_acc), max(all_acc)
    L = []
    prev = None
    for key, disp, _, _, family in EXP_B:
        if family != prev:
            if prev is not None:
                L.append(r"\addlinespace[2pt]")
            L.append(rf"\multicolumn{{9}}{{l}}{{{ifam(family)}"
                     rf"\textit{{{family}}}}} \\[1pt]")
            prev = family
        cells = []
        for model in MODEL_IDS:
            a = acc(model, key)
            cells.append(f"{heat(a, lo, hi)}{fmt(a)}")
        t = statistics.mean(TURNS[(MODEL_IDS[0], key)]) if TURNS[(MODEL_IDS[0], key)] else 0
        row_mean = statistics.mean(acc(m, key) for m in MODEL_IDS)
        L.append(f"{ib(key)} & {disp} & " + " & ".join(cells)
                 + f" & \\textbf{{{fmt(row_mean)}}} \\\\")
    body = "\n".join(L)
    heads = " & ".join(rf"\rotatebox{{60}}{{{d}}}" for _, d in MODELS)
    return rf"""
\begin{{table}}[t]
\centering
\caption{{\textbf{{Experiment B --- what it costs to deliver that evidence across turns.}}
Answer accuracy, $300$ questions $\times$ $3$ runs ($n=900$ per cell).
\textit{{Text}} conditions carry the question in words; \textit{{Visual}} conditions replace
it with layer renders; \textit{{Layered}} pairs each question shard with layers that do
(\textsc{{aligned}}) or do not (\textsc{{misaligned}}) support it; \textit{{Order}} permutes
evidence or shards. Shading is relative within the table.{legend(LEGEND_B)}
Conversations that overflowed the served context are scored incorrect rather than
dropped; Table~\ref{{tab:coverage}} reports how many.
Paired contrasts between these rows are in Table~\ref{{tab:contrasts}}.}}
\label{{tab:exp-b}}
\small
\setlength{{\tabcolsep}}{{5pt}}
\begin{{tabular}}{{ll cccccc c}}
\toprule
& & \multicolumn{{6}}{{c}}{{Answer accuracy}} & \\
\cmidrule(lr){{3-8}}
& Condition & {heads} & Mean \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def table_contrasts() -> str:
    L = []
    for name, arms, cond, ref, gloss in CONTRASTS:
        per = []
        for model in MODEL_IDS:
            d = acc(model, cond) - acc(model, ref)
            _, _, p = mcnemar(model, cond, ref)
            per.append(f"{signed(d)}{stars(p)}")
        pt, lo, hi, p, agree = pooled_contrast(cond, ref)
        ci = f"[{signed(lo)}, {signed(hi)}]"
        crosses = "yes" if lo * hi < 0 else "\\textbf{no}"
        L.append(f"{name} & {arms} & " + " & ".join(per)
                 + f" & \\textbf{{{signed(pt)}}} & {ci} & {agree}/6 \\\\")
    body = "\n".join(L)
    heads = " & ".join(rf"\rotatebox{{60}}{{{d}}}" for _, d in MODELS)
    return rf"""
\begin{{table}}[t]
\centering
\caption{{\textbf{{Paired contrasts.}} Each contrast compares two conditions on the
\emph{{same}} $300$ questions, so differences are computed per question and then
averaged rather than taken between independent means. Per-model significance is
McNemar over the $900$ paired observations
($^{{\ast}}p<.05$, $^{{\ast\ast}}p<.01$, $^{{\ast\ast\ast}}p<.001$).
\emph{{Pooled}} averages over all six models; its $95\%$ interval is a cluster
bootstrap resampling \emph{{questions}} (carrying all runs and models of a question
together), which is the conservative choice because the same questions recur
across models. \emph{{Agree}} counts models whose effect points the same way as the
pooled estimate. The top four rows are the paper's claims: all four have
intervals excluding zero and at least five of six models agreeing. The bottom
four have intervals spanning zero and are reported as nulls.}}
\label{{tab:contrasts}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{ll cccccc cc c}}
\toprule
& & \multicolumn{{6}}{{c}}{{Per-model $\Delta$ accuracy}} & \multicolumn{{2}}{{c}}{{Pooled}} & \\
\cmidrule(lr){{3-8}} \cmidrule(lr){{9-10}}
Contrast & Arms & {heads} & $\Delta$ & 95\% CI & Agree \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def table_turns() -> str:
    L = []
    for key, disp, _, _, family in EXP_B:
        cells = []
        for model in MODEL_IDS:
            t = TURNS.get((model, key), [])
            cells.append(f"{statistics.mean(t):.2f}" if t else "--")
        L.append(f"{ib(key)} & {disp} & " + " & ".join(cells) + r" \\")
    body = "\n".join(L)
    heads = " & ".join(rf"\rotatebox{{60}}{{{d}}}" for _, d in MODELS)
    return rf"""
\begin{{table}}[t]
\centering
\caption{{\textbf{{Assistant turns per conversation.}} Mean number of assistant
messages, pooled over three runs. Reported because accuracy alone cannot
distinguish a condition that is hard from one that is merely long: the
\textsc{{recap}} conditions add a turn by construction, and any judge sensitivity to
transcript length would therefore load onto exactly the contrasts in
Table~\ref{{tab:contrasts}} that involve them.}}
\label{{tab:turns}}
\small
\setlength{{\tabcolsep}}{{5pt}}
\begin{{tabular}}{{ll cccccc}}
\toprule
& Condition & {heads} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def table_coverage() -> str:
    """One row per affected (model, condition); complete models get one summary line."""
    keys = [k for k, *_ in EXP_A] + [k for k, *_ in EXP_B]
    label = {k: d for k, d, *_ in EXP_A}
    label.update({k: d for k, d, *_ in EXP_B})

    rows = []
    for model, disp in MODELS:
        affected = [k for k in keys if missing(model, k)]
        if not affected:
            full = f"{len(keys) * 300 * len(RUNS):,}".replace(",", "{,}")
            rows.append(rf"{disp} & \multicolumn{{5}}{{c}}{{\footnotesize "
                        rf"complete --- all ${full}$ conversations logged}} \\")
            continue
        for i, key in enumerate(affected):
            per = [len(DATASET_IDS[KEY_FN[key]]) - len(VERDICTS.get((r, model, key), {}))
                   for r in RUNS]
            head = disp if i == 0 else ""
            rows.append(f"{head} & {key}~{label[key]} & " + " & ".join(str(x) for x in per)
                        + f" & {sum(per)} \\\\")
        rows.append(r"\addlinespace[2pt]")
    body = "\n".join(rows).rstrip("\n").removesuffix(r"\addlinespace[2pt]")
    total = sum(missing(m, k) for m, _ in MODELS for k in keys)
    denom = len(MODELS) * len(keys) * 300 * len(RUNS)
    return rf"""
\begin{{table}}[t]
\centering
\caption{{\textbf{{Coverage audit.}} Conversations that never reached a verdict, per
replicate run. A prompt exceeding the served context window raises before a
verdict is written, so the sample vanishes from the log rather than scoring
zero; all tables above therefore score these ${total}$ conversations
(${100*total/denom:.1f}\%$ of ${denom:,}$) as \emph{{incorrect}} rather than dropping them.
The loss is concentrated entirely in \textsc{{InternVL3.5-8B}}, whose text tower
declares a $40{{,}}960$-token window that the longest conditions overflow --- and it
recurs at almost exactly the same size in all three runs, confirming it is
deterministic prompt length rather than sampling. Two consequences for reading
this draft: this model's \textsc{{misaligned}} contrast rests on a smaller and
systematically \emph{{longer}}-prompt subset than its \textsc{{aligned}} arm, and its
absolute accuracies are a lower bound. Re-serving it at a longer context before
camera-ready removes the caveat entirely.}}
\label{{tab:coverage}}
\small
\begin{{tabular}}{{ll rrr r}}
\toprule
& & \multicolumn{{3}}{{c}}{{Unlogged per run}} & \\
\cmidrule(lr){{3-5}}
Model & Condition & Final-0 & Final-1 & Final-2 & Total \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def table_stability() -> str:
    rows = []
    for model, disp in MODELS:
        sds = [acc_sd(model, k) for k, *_ in EXP_B]
        means = [acc(model, k) for k, *_ in EXP_B]
        rows.append(f"{disp} & {fmt(statistics.mean(means))} & {fmt(statistics.mean(sds),4)} "
                    f"& {fmt(max(sds),4)} & {sum(1 for s in sds if s > 0.03)}/14 \\\\")
    allsd = [acc_sd(m, k) for m in MODEL_IDS for k, *_ in EXP_B]
    body = "\n".join(rows)
    return rf"""
\begin{{table}}[t]
\centering
\caption{{\textbf{{Replicate stability.}} Final-0/1/2 use identical settings and
differ only in sampling (temperature $1.0$, no fixed seed). Across all
$6\times14=84$ model--condition cells the mean between-run standard deviation is
${statistics.mean(allsd):.4f}$ and the maximum is ${max(allsd):.4f}$ --- below the
$\approx0.028$ binomial standard error of a single $300$-question condition.
Sampling is therefore \emph{{not}} the dominant source of uncertainty in this
benchmark, which is why the contrasts in Table~\ref{{tab:contrasts}} are tested
per question rather than across replicates.}}
\label{{tab:stability}}
\small
\begin{{tabular}}{{l cccc}}
\toprule
Model & Mean acc. & Mean SD & Max SD & Cells SD $>.03$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def table_breakdown() -> str:
    """Accuracy by question category for the four load-bearing conditions."""
    keys = ["B2", "B6", "B10", "B11"]
    names = [r"\textsc{concat-q}", r"\textsc{concat-v}", r"\textsc{aligned}", r"\textsc{misaligned}"]
    cats = sorted({q["category"] for q in QMETA.values()})
    rows = []
    for cat in cats:
        ids = {t for t, q in QMETA.items() if q["category"] == cat}
        cells = []
        for key in keys:
            hits = tot = 0
            for model in MODEL_IDS:
                for run in RUNS:
                    v = scored(run, model, key)
                    for tid in ids & v.keys():
                        hits += v[tid]
                        tot += 1
            cells.append(fmt(hits / tot) if tot else "--")
        n = len(ids & DATASET_IDS[KEY_FN["B2"]])
        rows.append(f"\\texttt{{{cat.replace('_', r'\_')}}} & {n} & " + " & ".join(cells) + r" \\")
    body = "\n".join(rows)
    return rf"""
\begin{{table}}[t]
\centering
\caption{{\textbf{{Accuracy by question type}}, averaged over all six models and three
runs. Included because the aggregate contrasts hide where the cost falls:
the modality penalty is strikingly non-uniform. Moving from \textsc{{concat-q}} to
\textsc{{concat-v}} costs \emph{{attribute\_binding}} and \emph{{direct\_ask}} under $.09$,
but costs \emph{{counting}}, \emph{{occlusion}} and \emph{{spatial}} between $.25$ and $.28$
--- three to four times as much. Whatever the layer renders destroy, it is the
spatial and enumerative content rather than object attributes.}}
\label{{tab:by-category}}
\small
\begin{{tabular}}{{l r cccc}}
\toprule
Question type & $n$ & {" & ".join(names)} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


ICON_LIB = r"""% ---------------------------------------------------------------------------
% Condition glyphs.
%
% Each icon is a miniature diagram of the delivery schedule it labels, not a
% decorative pictograph. The vocabulary is fixed and worth stating once:
%
%   rounded bar          one conversation turn
%   bars touching        one turn carrying several shards (CONCAT family)
%   bars separated       one shard per turn (SHARDED family)
%   light -> dark ramp   shard order, first to last
%   square above a bar   a layer image attached to that turn
%   square offset        that layer landed on a turn it does not support
%
% Every distinction is carried by shape and position as well as by tone, so the
% glyphs survive greyscale printing and 8pt rendering.
% ---------------------------------------------------------------------------
\definecolor{icdark}{HTML}{2F6F9F}
\definecolor{icmid}{HTML}{86AECB}
\definecolor{iclite}{HTML}{CFE0EC}
\definecolor{icgrey}{HTML}{8A99A5}

\newlength{\icu}\setlength{\icu}{0.92ex}
\DeclareRobustCommand{\lvic}[1]{%
  \raisebox{-0.3ex}{\begin{tikzpicture}[x=\icu,y=\icu,line width=0.3pt,
      rounded corners=0.08]#1\end{tikzpicture}}\,}
\DeclareRobustCommand{\icbar}[4]{\fill[#3] (#1,#4) rectangle (#2,#4+0.8);}

% --- single-turn delivery ---------------------------------------------------
% FULL: the whole question arrives undivided.
\DeclareRobustCommand{\icFull}{\lvic{\icbar{0}{6}{icdark}{0}}}
% CONCAT: every shard present and in order, fused into one turn. Deliberately
% one continuous bar -- the only thing separating it from SHARDED is the
% absence of gaps, so drawing dividers here would destroy the contrast.
\DeclareRobustCommand{\icConcat}{\lvic{%
  \fill[iclite] (0,0) rectangle (2.05,0.8);
  \fill[icmid]  (1.95,0) rectangle (4.05,0.8);
  \fill[icdark] (3.95,0) rectangle (6,0.8);}}
% SHARDED: the same shards, one per turn.
\DeclareRobustCommand{\icSharded}{\lvic{%
  \icbar{0}{1.6}{iclite}{0}\icbar{2.2}{3.8}{icmid}{0}\icbar{4.4}{6}{icdark}{0}}}
% RECAP: sharded, then everything restated in one final turn.
\DeclareRobustCommand{\icRecap}{\lvic{%
  \icbar{0}{1.1}{iclite}{0}\icbar{1.5}{2.6}{icmid}{0}\icbar{3.0}{4.1}{icmid}{0}
  \fill[icdark] (4.6,-0.2) rectangle (6,1.0);}}
% SNOWBALL: every turn re-sends everything so far, so the turns grow.
\DeclareRobustCommand{\icSnowball}{\lvic{%
  \icbar{0}{1.0}{iclite}{0}\icbar{1.5}{3.2}{icmid}{0}\icbar{3.7}{6}{icdark}{0}}}
% SHUFFLED-CONCAT: one turn again, but the shard order is permuted.
\DeclareRobustCommand{\icShuffled}{\lvic{%
  \fill[icmid]  (0,0) rectangle (2.05,0.8);
  \fill[icdark] (1.95,0) rectangle (4.05,0.8);
  \fill[iclite] (3.95,0) rectangle (6,0.8);
  \draw[icgrey,line width=0.4pt,->] (1.0,1.2) .. controls (2.2,2.3) and (3.8,2.3) .. (5.0,1.2);}}

% --- turn plus attached layer image -----------------------------------------
% Squares sit directly on the bar so the pairing is legible at 8pt: touching
% means supported, offset means not.
\DeclareRobustCommand{\icthree}{%
  \icbar{0}{1.6}{iclite}{0}\icbar{2.2}{3.8}{icmid}{0}\icbar{4.4}{6}{icdark}{0}}
% ALIGNED: every layer rests on the shard it supports -- squares seated on bars,
% nothing to trace.
\DeclareRobustCommand{\icAligned}{\lvic{\icthree
  \fill[icmid] (0,0.95) rectangle (1.6,1.95);
  \fill[icmid] (2.2,0.95) rectangle (3.8,1.95);
  \fill[icmid] (4.4,0.95) rectangle (6,1.95);}}
% MISALIGNED: identical layers and identical per-turn counts, but each one is
% lifted off its own shard and tied to another. The crossing IS the condition.
\DeclareRobustCommand{\icMisaligned}{\lvic{\icthree
  \fill[icmid] (0,1.65) rectangle (1.6,2.65);
  \fill[icmid] (2.2,1.65) rectangle (3.8,2.65);
  \fill[icmid] (4.4,1.65) rectangle (6,2.65);
  \draw[icgrey,line width=0.5pt] (0.8,1.6)--(5.2,0.88);
  \draw[icgrey,line width=0.5pt] (5.2,1.6)--(0.8,0.88);}}
% EVIDENCE-FIRST: every layer bunched onto the opening turn.
\DeclareRobustCommand{\icEvFirst}{\lvic{\icthree
  \fill[icmid] (0,0.95) rectangle (0.5,1.95);
  \fill[icmid] (0.55,0.95) rectangle (1.05,1.95);
  \fill[icmid] (1.1,0.95) rectangle (1.6,1.95);}}
% EVIDENCE-LAST: every layer withheld until the closing turn.
\DeclareRobustCommand{\icEvLast}{\lvic{\icthree
  \fill[icmid] (4.4,0.95) rectangle (4.9,1.95);
  \fill[icmid] (4.95,0.95) rectangle (5.45,1.95);
  \fill[icmid] (5.5,0.95) rectangle (6,1.95);}}

% --- Experiment A: what evidence is on the table at all ---------------------
% All three share a 3.2 x 2.7 bounding box so they read as equally weighted.
\DeclareRobustCommand{\icRGB}{\lvic{%
  \draw[icdark,fill=iclite,line width=0.45pt] (0,0) rectangle (3.2,2.7);
  \fill[icdark] (0.72,1.95) circle (0.34);
  \fill[icmid] (0.3,0.25) -- (1.6,1.55) -- (2.9,0.25) -- cycle;}}
% ALL-LAYERS: every layer offered, the model must choose among them.
\DeclareRobustCommand{\icAll}{\lvic{%
  \draw[icgrey,fill=white,line width=0.35pt] (0,0) rectangle (2.1,1.75);
  \draw[icmid,fill=iclite,line width=0.35pt] (0.55,0.48) rectangle (2.65,2.23);
  \draw[icdark,fill=icmid,line width=0.45pt] (1.1,0.95) rectangle (3.2,2.7);}}
% ORACLE: only the gold minimal sufficient set is handed over; the rest is
% drawn as absent rather than merely unselected.
\DeclareRobustCommand{\icOracle}{\lvic{%
  \draw[icgrey,dash pattern=on 0.45 off 0.4,line width=0.3pt] (0,0) rectangle (2.1,1.75);
  \draw[icgrey,dash pattern=on 0.45 off 0.4,line width=0.3pt] (0.55,0.48) rectangle (2.65,2.23);
  \draw[icdark,fill=icdark,line width=0.45pt] (1.1,0.95) rectangle (3.2,2.7);}}

% --- family markers ---------------------------------------------------------
\DeclareRobustCommand{\icText}{\lvic{%
  \fill[icdark] (0,1.75) rectangle (3.4,2.15);
  \fill[icmid] (0,1.0) rectangle (2.6,1.4);
  \fill[iclite] (0,0.25) rectangle (3.1,0.65);}}
\DeclareRobustCommand{\icVisual}{\lvic{%
  \draw[icdark,fill=iclite,line width=0.45pt] (0,0) rectangle (3.0,2.4);
  \fill[icdark] (0.68,1.72) circle (0.3);
  \fill[icmid] (0.28,0.22) -- (1.5,1.4) -- (2.72,0.22) -- cycle;}}
\DeclareRobustCommand{\icLayered}{\lvic{%
  \draw[icmid,fill=iclite,line width=0.35pt] (0,0) rectangle (2.0,1.6);
  \draw[icdark,fill=icmid,line width=0.45pt] (0.9,0.8) rectangle (2.9,2.4);}}
\DeclareRobustCommand{\icOrder}{\lvic{%
  \fill[iclite] (0,0) rectangle (1.5,0.8);
  \fill[icdark] (1.9,0) rectangle (3.4,0.8);
  \fill[icmid] (3.8,0) rectangle (5.3,0.8);
  \draw[icgrey,line width=0.4pt,->] (0.75,1.2) .. controls (1.8,2.3) and (3.6,2.3) .. (4.55,1.2);}}
"""

PREAMBLE_TMPL = r"""% ---------------------------------------------------------------------------
% Layered-VQA -- result tables
% GENERATED FILE. Do not hand-edit: rerun `python overleaf/make_tables.py`.
%
% Source logs : vendor/conv-smi/logs/Final-{0,1,2}
% Convention  : unlogged conversations are scored INCORRECT (see tab:coverage)
%
% To drop these into a paper, copy everything between the BEGIN/END markers and
% add to your own preamble:
%     \usepackage{booktabs, colortbl, xcolor, graphicx, rotating}
%     \definecolor{tabhi}{HTML}{2F6F9F}
% {PKGNOTE}
% ---------------------------------------------------------------------------
\documentclass[10pt]{article}
\usepackage[margin=1in,landscape]{geometry}
\usepackage{lmodern}   % OT1/cmr has no bold small-caps; lmodern does
\usepackage[T1]{fontenc}
\usepackage{booktabs}
\usepackage{colortbl}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{rotating}
{TIKZ}
\usepackage{amssymb}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf}

\definecolor{tabhi}{HTML}{2F6F9F}

{ICON_LIB}
\begin{document}
\section*{Layered-VQA: result tables (v0 draft)}
\label{sec:limitations}

\noindent
All numbers are pooled over three identically-configured replicate runs
(\texttt{Final-0/1/2}) of the same $300$-question benchmark.
Every conversation was graded by the model under test acting as its own judge;
see the coverage and stability tables for the two caveats that follow from the
logs themselves.

% ======================== BEGIN TABLES ========================
"""

POSTAMBLE = r"""
% ========================= END TABLES =========================
\end{document}
"""


def preamble() -> str:
    """The variant's preamble. The glyph library and its TikZ dependency are
    omitted entirely from the plain build, so that file drops into a paper
    whose preamble has no TikZ."""
    return (PREAMBLE_TMPL
            .replace("{TIKZ}", "\\usepackage{tikz}\n\\usetikzlibrary{calc}" if ICONS else "")
            .replace("{ICON_LIB}", ICON_LIB if ICONS else "")
            .replace("{PKGNOTE}", "    plus \\usepackage{tikz} for the condition glyphs"
                     if ICONS else "    (no TikZ needed: this build has no glyphs)"))


def build(path: Path) -> None:
    parts = [
        preamble(),
        table_exp_a(),
        table_grounding(),
        r"\clearpage",
        table_exp_b(),
        table_contrasts(),
        r"\clearpage",
        table_turns(),
        table_by := table_breakdown(),
        r"\clearpage",
        table_coverage(),
        table_stability(),
        POSTAMBLE,
    ]
    path.write_text("\n".join(parts), encoding="utf-8")
    print(f"  wrote {path.name}"
          f"  ({'with glyphs' if ICONS else 'plain, no TikZ'})")


def main() -> None:
    global ICONS
    print(f"models {len(MODELS)} | runs {len(RUNS)} | "
          f"A cells {len(MODELS)*len(EXP_A)} | B cells {len(MODELS)*len(EXP_B)}")
    keys = [k for k, *_ in EXP_A] + [k for k, *_ in EXP_B]
    lost = sum(missing(m, k) for m, _ in MODELS for k in keys)
    print(f"scored-wrong (missing) conversations: {lost}")

    ICONS = True
    build(OUT)
    ICONS = False
    build(OUT_PLAIN)


if __name__ == "__main__":
    main()
