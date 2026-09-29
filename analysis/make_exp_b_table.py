"""Experiment-B main table: consensus final-turn accuracy, shaded by the change
from each model's own FULL baseline.

Reads the consensus CSV (majority vote of all post-hoc judges, per conversation)
print("%--- means-only variant written to tab_exp_b_plain.tex")
"""
import csv, statistics as st, sys
from collections import defaultdict
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "consensus.csv")
OUT = Path(__file__).parent

EXP_B = [
    ("B1", r"\textsc{full}",            "full",              "layered_vqa_text.json",                   "Text"),
    ("B2", r"\textsc{concat}",          "concat",            "layered_vqa_text.json",                   "Text"),
    ("B3", r"\textsc{sharded}",         "sharded",           "layered_vqa_text.json",                   "Text"),
    ("B4", r"\textsc{recap}",           "recap-concat",      "layered_vqa_text.json",                   "Text"),
    ("B5", r"\textsc{snowball}",        "snowball",          "layered_vqa_text.json",                   "Text"),
    ("B6", r"\textsc{concat}",          "full",              "layered_vqa_visual_canonical.json",       "Visual"),
    ("B7", r"\textsc{sharded}",         "sharded-scripted",  "layered_vqa_visual_canonical.json",       "Visual"),
    ("B8", r"\textsc{recap}",           "recap-concat",      "layered_vqa_visual_canonical.json",       "Visual"),
    ("B9", r"\textsc{snowball}",        "snowball-scripted", "layered_vqa_visual_canonical.json",       "Visual"),
    ("B10", r"\textsc{aligned}",        "sharded-scripted",  "layered_vqa_aligned_vq.json",             "Layered"),
    ("B11", r"\textsc{misaligned}",     "sharded-scripted",  "layered_vqa_misaligned_vq.json",          "Layered"),
    ("B12", r"\textsc{ev-first}",       "sharded-scripted",  "layered_vqa_visual_evidence_first.json",  "Order"),
    ("B13", r"\textsc{ev-last}",        "sharded-scripted",  "layered_vqa_visual_evidence_last.json",   "Order"),
    ("B14", r"\textsc{shuffled}",       "shuffle-concat",    "layered_vqa_text.json",                   "Order"),
]

DISPLAY = {
    "qwen2.5-vl-3b":          (r"Qwen2.5-VL-3B", "3B"),
    "qwen3-vl-4b":            (r"Qwen3-VL-4B", "4B"),
    "qwen2.5-vl-7b":          (r"Qwen2.5-VL-7B", "7B"),
    "qwen3-vl-8b":            (r"Qwen3-VL-8B", "8B"),
    "internvl3.5-8b":         (r"InternVL3.5-8B", "8B"),
    "ministral3-8b-instruct": (r"Ministral-3-8B", "8B"),
    "qwen3.5-9b":             (r"Qwen3.5-9B", "9B"),
    "pixtral-12b":            (r"Pixtral-12B", "12B"),
    "qwen3.6-27b":            (r"Qwen3.6-27B", "27B"),
    "qwen3-vl-30b-a3b":       (r"Qwen3-VL-30B", "30B"),
    "qwen2.5-vl-32b":         (r"Qwen2.5-VL-32B", "32B"),
    # Closed weights, through OpenRouter. Their cells are NOT consensus
    # verdicts: the judge-by-assistant matrix was never filled for them, so
    # each run is graded by the model itself and the cell is the mean over its
    # own runs. gpt-5.6-luna has three runs, gemini-3.7-flash two, so the SD
    # under a gemini cell is over two values and is a range, not an estimate.
    "gpt-5.6-luna":           (r"GPT-5.6-Luna$^{\dagger}$", "API"),
    "gemini-3.7-flash":       (r"Gemini-3.7-Flash$^{\dagger}$", "API"),
}

# --- load -------------------------------------------------------------------
per_run = defaultdict(dict)
for row in csv.DictReader(SRC.open(encoding="utf-8")):
    key = (row["model"], row["conv_type"], row["dataset_fn"])
    per_run[key][row["run"]] = int(row["final"]) / int(row["n"])

SPEC = {k: (ct, fn) for k, _, ct, fn, _ in EXP_B}

def stats(model, key):
    vals = list(per_run[(model,) + SPEC[key]].values())
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)

MODELS = sorted(DISPLAY, key=lambda m: stats(m, "B1")[0])
KEYS = [k for k, *_ in EXP_B]

# --- shading ----------------------------------------------------------------
# Diverging, anchored on each model's own FULL score: a cell says "what this
# delivery schedule cost *this* model", which is the comparison the paper makes.
# One-sided red would throw away the (rarer, but real) cases where a schedule
# helps -- RECAP beats FULL for 8 of 11 models.
DOWN_CAP, UP_CAP, DOWN_MAX, UP_MAX = 0.50, 0.15, 72, 58

def shade(delta):
    if delta < 0:
        pct = round(DOWN_MAX * min(1.0, -delta / DOWN_CAP))
        return r"\cellcolor{lvqaDown!%d}" % pct
    pct = round(UP_MAX * min(1.0, delta / UP_CAP))
    return r"\cellcolor{lvqaUp!%d}" % pct


def num(x):
    """Leading-zero-less, as is conventional for accuracies in [0,1]."""
    return f"{x:.3f}".lstrip("0")


def cell(model, key, base, with_sd):
    """One cell: accuracy, optionally with the run-to-run SD stacked beneath it.

    Stacked rather than inline -- 14 conditions of ".729 +-.02" side by side comes
    to 738pt, nearly twice the ICLR text block, while stacking keeps the table at
    its means-only width and spends vertical space instead.
    """
    mean, sd = stats(model, key)
    body = num(mean)
    if key == "B1":
        body = r"\textbf{%s}" % body
    if with_sd:
        body = r"\stk{%s}{%s}" % (body, f"{sd:.2f}".lstrip("0"))
    if key == "B1":
        return r"\cellcolor{lvqaBase!30}" + body
    return shade(mean - base) + body


# Retention: the share of the FULL score that survives a family of schedules.
FAMILIES = [("Text", ["B2", "B3", "B4", "B5"]),
            ("Visual", ["B6", "B7", "B8", "B9"]),
            ("Layered", ["B10", "B11", "B12", "B13"])]


def retention(model, keys):
    base = stats(model, "B1")[0]
    return 100.0 * st.mean([stats(model, k)[0] for k in keys]) / base


def ret_cell(value):
    """Same diverging scale, in retention units, so the block reads with the grid."""
    return shade((value - 100.0) / 100.0 * 0.6) + f"{value:.0f}"


# --- emit -------------------------------------------------------------------
def build(with_sd):
    colspec = "l " + "c" * len(KEYS) + " ccc"
    L = []
    A = L.append
    A(r"\begin{tabular}{" + colspec + "}")
    A(r"\toprule")
    A(r"& \multicolumn{1}{c}{\textsc{base}} & \multicolumn{4}{c}{\textsc{question in text}}"
      r" & \multicolumn{4}{c}{\textsc{question in layers}}"
      r" & \multicolumn{2}{c}{\textsc{alignment}} & \multicolumn{3}{c}{\textsc{order}}"
      r" & \multicolumn{3}{c}{\textsc{retained \%}} \\")
    A(r"\cmidrule(lr){2-2}\cmidrule(lr){3-6}\cmidrule(lr){7-10}"
      r"\cmidrule(lr){11-12}\cmidrule(lr){13-15}\cmidrule(lr){16-18}")
    heads = " & ".join(r"\rot{%s}" % d for _, d, *_ in EXP_B)
    A(r"Model & " + heads + r" & \rot{Text} & \rot{Visual} & \rot{Layered} \\")
    A(r"\midrule")
    for m in MODELS:
        base = stats(m, "B1")[0]
        name, size = DISPLAY[m]
        cells = " & ".join(cell(m, k, base, with_sd) for k in KEYS)
        rets = " & ".join(ret_cell(retention(m, ks)) for _, ks in FAMILIES)
        A(f"{name} & {cells} & {rets} \\\\")
    A(r"\midrule")
    means = [st.mean([stats(m, k)[0] for m in MODELS]) for k in KEYS]
    b1 = means[0]
    row = " & ".join(
        (r"\cellcolor{lvqaBase!30}\textbf{%s}" % num(v)) if i == 0 else shade(v - b1) + num(v)
        for i, v in enumerate(means))
    rets = " & ".join(ret_cell(st.mean([retention(m, ks) for m in MODELS])) for _, ks in FAMILIES)
    A(r"\textit{Mean} & " + row + " & " + rets + r" \\")
    dlt = " & ".join(["---"] + [f"{v - b1:+.3f}".replace("0.", ".") for v in means[1:]])
    A(r"\textit{$\Delta$ vs \textsc{full}} & " + dlt + r" & & & \\")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    return "\n".join(L)


PRE = r"""% requires: booktabs, xcolor (table option), graphicx
\definecolor{lvqaDown}{RGB}{202,74,56}    % a schedule costs accuracy
\definecolor{lvqaUp}{RGB}{47,124,140}     % a schedule recovers accuracy
\definecolor{lvqaBase}{RGB}{120,120,120}  % the FULL reference column
\newcommand{\rot}[1]{\rotatebox{90}{\scriptsize #1}}
\newcommand{\stk}[2]{\begin{tabular}[c]{@{}c@{}}#1\\[-3.5pt]{\tiny\textcolor{black!55}{$\pm$#2}}\end{tabular}}
"""
(OUT / "tab_exp_b.tex").write_text(PRE + build(True), encoding="utf-8")
(OUT / "tab_exp_b_plain.tex").write_text(PRE + build(False), encoding="utf-8")
print(PRE + build(True))
print("%--- means-only variant written to tab_exp_b_plain.tex")
