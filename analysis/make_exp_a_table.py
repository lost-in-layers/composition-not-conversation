"""Experiment-A main table: what the models can do when the evidence is handed
to them, and whether they can say which evidence they used.

Merges the two former tables. They were always meant to be read together -- the
old grounding caption told the reader to compare against the accuracy table --
and splitting them let a model look strong on judge-scored accuracy while being
weak on the judge-free evidence columns two pages away.

Accuracy is consensus final-turn (majority vote of all eleven post-hoc judges),
matching the Experiment-B table. The evidence columns are computed
programmatically from the predicted layer set against the annotation; no judge
is involved. Joint is recomputed against the same consensus verdict so the two
halves of the table cannot disagree about correctness.

Layout note: every numeric column is one fixed width and every body cell is the
same two lines tall. Mixing stacked accuracy cells with single-line evidence
cells makes the shaded blocks ragged -- the colour then reads as noise rather
than as a scale, which is the whole point of shading it.
"""
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

SCRATCH = Path(sys.argv[1])
OUT = Path(__file__).parent

EXP_A = [("A1", "full", "layered_vqa_text.json"),
         ("A2", "full", "layered_vqa_grounding.json"),
         ("A3", "full", "layered_vqa_oracle_gold.json")]

DISPLAY = {
    "qwen2.5-vl-3b": "Qwen2.5-VL-3B", "qwen3-vl-4b": "Qwen3-VL-4B",
    "qwen2.5-vl-7b": "Qwen2.5-VL-7B", "qwen3-vl-8b": "Qwen3-VL-8B",
    "internvl3.5-8b": "InternVL3.5-8B", "ministral3-8b-instruct": "Ministral-3-8B",
    "qwen3.5-9b": "Qwen3.5-9B", "pixtral-12b": "Pixtral-12B",
    "qwen3.6-27b": "Qwen3.6-27B", "qwen3-vl-30b-a3b": "Qwen3-VL-30B",
    "qwen2.5-vl-32b": "Qwen2.5-VL-32B",
    # Closed weights. Their accuracy columns are self-judged, not consensus --
    # see the Experiment-B table's dagger note. The evidence columns are
    # judge-free either way, so those are directly comparable.
    "gpt-5.6-luna": r"GPT-5.6-Luna$^{\dagger}$",
    "gemini-3.7-flash": r"Gemini-3.7-Flash$^{\dagger}$",
}

RUNS = ["Final-0", "Final-1", "Final-2"]

# label, field, higher-is-better
GCOLS = [("P", "precision", True), ("R", "recall", True), ("F1", "f1", True),
         ("EM", "exact_match", True), ("Comp.", "complete", True),
         ("Exc.", "excess_rate", False), ("Dist.", "distractor_captured", False),
         ("Inv.", "invalid", False), ("Joint", "joint_consensus", True)]

# --- load -------------------------------------------------------------------
per_run = defaultdict(dict)
for row in csv.DictReader((SCRATCH / "consensus.csv").open(encoding="utf-8")):
    per_run[(row["model"], row["conv_type"], row["dataset_fn"])][row["run"]] = \
        int(row["final"]) / int(row["n"])

ground = defaultdict(dict)
for row in csv.DictReader((SCRATCH / "grounding.csv").open(encoding="utf-8")):
    ground[(row["model"], row["run"])] = {k: float(v) for k, v in row.items()
                                          if k not in ("run", "model")}
for row in csv.DictReader((SCRATCH / "joint.csv").open(encoding="utf-8")):
    ground[(row["model"], row["run"])]["joint_consensus"] = float(row["joint_consensus"])

SPEC = {k: (ct, fn) for k, ct, fn in EXP_A}


def stats(model, key):
    v = list(per_run[(model,) + SPEC[key]].values())
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def gstats(model, field):
    # Whatever runs exist for this model, rather than the fixed open-model
    # RUNS list: the closed models live in Final-Closed-* and one of them has
    # two runs, not three.
    v = [d[field] for (m, _r), d in ground.items() if m == model]
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


MODELS = sorted(DISPLAY, key=lambda m: stats(m, "A1")[0])

# --- shading ----------------------------------------------------------------
DOWN_CAP, UP_CAP, DOWN_MAX, UP_MAX = 0.50, 0.15, 72, 58
RAMP_DOWN, RAMP_UP = 60, 52   # evidence columns, mean-anchored
B = chr(92)


def cc(color, pct):
    return B + "cellcolor{" + color + "!" + str(pct) + "}"


def shade(delta):
    """Accuracy shading, anchored on the model's own RGB baseline."""
    if delta < 0:
        return cc("lvqaDown", round(DOWN_MAX * min(1.0, -delta / DOWN_CAP)))
    return cc("lvqaUp", round(UP_MAX * min(1.0, delta / UP_CAP)))


def ramp(value, mean, span, higher_is_better):
    """Diverging ramp for the evidence columns, anchored on the column mean.

    Two things this fixes over a min-max ramp.

    White now sits on a number the reader can check -- the column mean, which is
    printed in the Mean row. Under min-max the neutral point was the midpoint of
    the observed range, so a model at the field average could still be shaded
    red simply because the worst model was further from the mean than the best.

    And the scale is continuous through zero. The old ramp jumped from red!1 to
    teal!26 across the halfway mark, so two cells six thousandths apart -- .168
    and .174 in Ex Match -- read as different categories.

    An outlier still sets the intensity, through `span`, but no longer the sign.
    """
    d = value - mean
    if not higher_is_better:
        d = -d
    t = 0.0 if span <= 0 else max(-1.0, min(1.0, d / span))
    if t < 0:
        return cc("lvqaDown", round(RAMP_DOWN * -t))
    return cc("lvqaUp", round(RAMP_UP * t))


# --- cell formatting --------------------------------------------------------
def num(x):
    return f"{x:.3f}".lstrip("0")


def signed(x):
    return f"{x:+.3f}".replace("0.", ".")


def sd2(x):
    return f"{x:.2f}".lstrip("0")


def stk(body, sd):
    return B + "stk{" + body + "}{" + sd2(sd) + "}"


def stkb(body):
    """Same two-line box with an empty second line, to keep rows uniform."""
    return B + "stkb{" + body + "}"


def bf(s):
    return B + "textbf{" + s + "}"


def sc(s):
    return B + "textsc{" + s + "}"


def mc(n, body):
    return B + "multicolumn{" + str(n) + "}{c}{" + body + "}"


def build():
    """The 12-column layout used in the paper: accuracy, evidence, errors, joint."""
    stat = {}
    for _lab, f, _h in GCOLS:
        vals = [gstats(m, f)[0] for m in MODELS]
        mu = st.mean(vals)
        stat[f] = (mu, max(abs(v - mu) for v in vals))

    NL = " " + B + B
    L = [B + "begin{tabular}{@{}l*{12}{Z}@{}}", B + "toprule"]
    L.append("& " + B + "multicolumn{3}{c}{" + B + "grp{answer accuracy}}"
             + " & " + B + "multicolumn{5}{c}{" + B + "grp{evidence selection}}"
             + " & " + B + "multicolumn{3}{c}{" + B + "grp{error modes}}"
             + " & " + B + "multicolumn{1}{c}{" + B + "grp{joint}}" + NL)
    L.append(B + "cmidrule(lr){2-4}" + B + "cmidrule(lr){5-9}"
             + B + "cmidrule(lr){10-12}" + B + "cmidrule(l){13-13}")
    heads = [sc("rgb"), sc("all-lyr"), sc("oracle"), "Precision", "Recall",
             "F1-score", "Ex Match", "Comp.", "Exc.", "Dist.", "Inv.", "Acc."]
    L.append("Model & " + " & ".join(
        B + "multicolumn{1}{c}{" + B + "rot{" + h + "}}" for h in heads) + NL)
    L.append(B + "midrule")

    for m in MODELS:
        base = stats(m, "A1")[0]
        cells = []
        for key, *_ in EXP_A:
            mean, sd = stats(m, key)
            body = bf(num(mean)) if key == "A1" else num(mean)
            pre = cc("lvqaBase", 30) if key == "A1" else shade(mean - base)
            cells.append(pre + stk(body, sd))
        for _lab, field, hib in GCOLS:
            v, sd = gstats(m, field)
            cells.append(ramp(v, *stat[field], hib) + stk(num(v), sd))
        L.append(DISPLAY[m] + " & " + " & ".join(cells) + NL)

    L.append(B + "midrule")
    mu = {k: st.mean([stats(m, k)[0] for m in MODELS]) for k, *_ in EXP_A}
    row = [cc("lvqaBase", 30) + bf(num(mu["A1"]))]
    row += [shade(mu[k] - mu["A1"]) + num(mu[k]) for k in ("A2", "A3")]
    # The Mean row is the anchor the evidence columns are shaded against, so it
    # is deliberately left unshaded -- colouring it would shade it against itself.
    row += [num(stat[f][0]) for _l, f, _h in GCOLS]
    L.append(B + "textit{Mean} & " + " & ".join(row) + NL)

    L += [B + "bottomrule", B + "end{tabular}"]
    return chr(10).join(L)


PREAMBLE = "\n".join([
    "% requires: booktabs and xcolor (table option). Deliberately NOT array:",
    "% the uniform cell width comes from a parbox rather than a >{}p{} column,",
    "% which keeps this working where array and colortbl internals disagree.",
    B + "definecolor{lvqaDown}{RGB}{202,74,56}",
    B + "definecolor{lvqaUp}{RGB}{47,124,140}",
    B + "definecolor{lvqaBase}{RGB}{120,120,120}",
    B + "newlength{" + B + "lvqacw}",
    B + "setlength{" + B + "lvqacw}{8.4mm}",
    "% Every body cell is this box: one fixed width, two lines tall. The uniform",
    "% width is what makes the shading read as a grid instead of a patchwork.",
    B + "newcommand{" + B + "stk}[2]{" + B + "parbox[c]{" + B + "lvqacw}{" + B
    + "centering #1" + B + B + "[-3pt]{" + B + "tiny" + B
    + "textcolor{black!55}{$" + B + "pm$#2}}}}",
    B + "newcommand{" + B + "stkb}[1]{" + B + "parbox[c]{" + B + "lvqacw}{" + B
    + "centering #1" + B + B + "[-3pt]{" + B + "tiny" + B + "phantom{0}}}}",
    "",
])

(OUT / "tab_exp_a.tex").write_text(PREAMBLE + build() + "\n", encoding="utf-8")
print(PREAMBLE + build())
