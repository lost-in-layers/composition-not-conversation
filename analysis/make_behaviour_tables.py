"""The behavioural tables, in the same visual system as the two main tables.

Three tables:

  tab_firstattempt   accuracy against how early in the conversation the model
                     first commits to an answer. Theirs bins by conversation
                     progress and stops there; ours adds the share of those
                     early answers that were ungrounded -- given before any
                     gold layer had arrived -- which is what makes an early
                     commitment a mistake rather than a preference.

  tab_verbosity      accuracy against the model's own response length, in
                     quintiles computed inside each (model, condition) so the
                     bins compare a model against itself. Split by whether the
                     evidence arrived in one turn or many, because that is the
                     comparison our benchmark is for.

  tab_ungrounded     ours alone. What share of correct answers were produced
                     before the evidence needed to support them was delivered.

Usage: python overleaf/make_behaviour_tables.py <csv-dir>
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

SRC = Path(sys.argv[1])
OUT = Path(__file__).parent
B = chr(92)

# The eleven open models set every shading scale and every mean in these
# tables, so adding the closed models below changes no number the paper
# already cites. The closed models are placed on the open field's scale
# rather than redefining it.
OPEN = ["qwen2.5-vl-3b", "qwen3-vl-4b", "qwen2.5-vl-7b", "qwen3-vl-8b",
        "internvl3.5-8b", "ministral3-8b-instruct", "qwen3.5-9b", "pixtral-12b",
        "qwen3.6-27b", "qwen3-vl-30b-a3b", "qwen2.5-vl-32b"]
CLOSED = ["gpt-5.6-luna", "gemini-3.7-flash"]
ORDER = OPEN + CLOSED
DISPLAY = {
    "qwen2.5-vl-3b": "Qwen2.5-VL-3B", "qwen3-vl-4b": "Qwen3-VL-4B",
    "qwen2.5-vl-7b": "Qwen2.5-VL-7B", "qwen3-vl-8b": "Qwen3-VL-8B",
    "internvl3.5-8b": "InternVL3.5-8B", "ministral3-8b-instruct": "Ministral-3-8B",
    "qwen3.5-9b": "Qwen3.5-9B", "pixtral-12b": "Pixtral-12B",
    "qwen3.6-27b": "Qwen3.6-27B", "qwen3-vl-30b-a3b": "Qwen3-VL-30B",
    "qwen2.5-vl-32b": "Qwen2.5-VL-32B",
    "gpt-5.6-luna": "GPT-5.6-Luna", "gemini-3.7-flash": "Gemini-3.7-Flash",
}
FAMILY = {"B1": "Text", "B2": "Text", "B3": "Text", "B4": "Text", "B5": "Text",
          "B6": "Visual", "B7": "Visual", "B8": "Visual", "B9": "Visual",
          "B10": "Layered", "B11": "Layered", "B12": "Order", "B13": "Order",
          "B14": "Order"}


def rows(name):
    return list(csv.DictReader((SRC / name).open(encoding="utf-8")))


def cc(color, pct):
    return B + "cellcolor{" + color + "!" + str(int(pct)) + "}"


RAMP_DOWN, RAMP_UP = 60, 52


def anchor(values):
    """Neutral point and half-width for one shading scope.

    Returns the mean, plus the largest deviation from it, so the ramp can be
    continuous through white at a value the reader can look up in the Mean row.
    """
    if not values:
        return 0.0, 0.0
    mu = sum(values) / len(values)
    return mu, max((abs(v - mu) for v in values), default=0.0)


def ramp(value, mean, span, higher_is_better=True):
    """Diverging ramp anchored on `mean`, continuous through white.

    Replaces a min-max ramp that had two problems. Its neutral point was the
    midpoint of the observed range rather than any real quantity, and it jumped
    straight from red!1 to teal!26 at that midpoint, so two cells a thousandth
    apart could read as different categories.
    """
    d = value - mean
    if not higher_is_better:
        d = -d
    t = 0.0 if span <= 0 else max(-1.0, min(1.0, d / span))
    if t < 0:
        return cc("lvqaDown", round(RAMP_DOWN * -t))
    return cc("lvqaUp", round(RAMP_UP * t))


def num(x):
    """Value in a fixed-width box.

    Without it the widest column header stretches one column and the shaded
    blocks stop lining up -- the same raggedness the main tables fix with
    their stacked cell. Here a plain box is enough: no cell has a second line.
    """
    return B + "nb{" + f"{x:.3f}".lstrip("0") + "}"


def group(label, ncols):
    """A run-in heading inside the body, so the two closed models read as a
    separate field rather than as two more entries in the open ranking."""
    return (B + "addlinespace[2pt]" + B + "multicolumn{" + str(ncols)
            + "}{@{}l}{" + B + "itshape " + label + "} " + B + B
            + B + "addlinespace[1pt]")


def wrap(body, colspec, header_lines):
    L = [B + "begin{tabular}{" + colspec + "}", B + "toprule"]
    L += header_lines
    L += [B + "midrule"] + body + [B + "bottomrule", B + "end{tabular}"]
    return "\n".join(L)


PREAMBLE = "\n".join([
    "% requires: booktabs and xcolor (table option).",
    "% Same three colours as the two main tables; drop these lines if already loaded.",
    B + "definecolor{lvqaDown}{RGB}{202,74,56}",
    B + "definecolor{lvqaUp}{RGB}{47,124,140}",
    B + "definecolor{lvqaBase}{RGB}{120,120,120}",
    "% one width for every numeric cell, so the shading lines up into a grid",
    B + "providecommand{" + B + "nb}[1]{" + B + "makebox[8.4mm][c]{#1}}",
    "",
])


# ------------------------------------------------------------------ table 1 --
def first_attempt():
    BINS = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    LAB = ["earliest", "early", "midway", "late", "latest"]
    acc = {}
    open_tot = defaultdict(lambda: [0, 0])
    for r in rows("beh_bins.csv"):
        n, c = int(r["n"]), int(r["correct"])
        acc[(r["model"], r["bin"])] = (n, c)
        if r["model"] in OPEN:
            open_tot[r["bin"]][0] += n
            open_tot[r["bin"]][1] += c
    body = [group("Open weights", 8)]
    for m in ORDER:
        if m == CLOSED[0]:
            body.append(group("API only", 8))
        # Anchored on the model's own mean across the five bins. Shading against
        # the whole table made colour report which model a row belongs to -- the
        # weakest row came out uniformly red -- and buried the shape across the
        # bins, which is the only thing this table is for.
        mu, span = anchor([c / n for b in BINS
                           for (n, c) in [acc.get((m, b), (0, 0))] if n])
        cells = []
        for b in BINS:
            n, c = acc.get((m, b), (0, 0))
            cells.append(ramp(c / n, mu, span) + num(c / n) if n else "---")
        # Where the model commits most often -- the mode of its own distribution.
        counts = [acc.get((m, b), (0, 0))[0] for b in BINS]
        mode = LAB[counts.index(max(counts))] if any(counts) else "---"
        share = max(counts) / max(1, sum(counts))
        body.append(DISPLAY[m] + " & " + " & ".join(cells)
                    + f" & {B}textit{{{mode}}} & {share*100:.0f}{B}% " + B + B)
    means = [open_tot[b][1] / open_tot[b][0] for b in BINS]
    body.append(B + "midrule")
    body.append(B + "textit{Mean, open} & " + " & ".join(num(v) for v in means)
                + " & & " + B + B)

    head = ["& " + B + "multicolumn{5}{c}{" + B
            + "textsc{first answer attempt}} & "
            + B + "multicolumn{2}{c}{" + B + "textsc{commits}} " + B + B,
            B + "cmidrule(lr){2-6}" + B + "cmidrule(lr){7-8}",
            "Model & " + " & ".join(f"{l}" for l in LAB)
            + " & mode & share " + B + B,
            "& " + " & ".join("{" + B + "tiny " + b + B + "%}" for b in BINS)
            + " & & " + B + B]
    tex = PREAMBLE + wrap(body, "l ccccc cc", head)
    (OUT / "tab_firstattempt.tex").write_text(tex + "\n", encoding="utf-8")
    print("  tab_firstattempt.tex")
    return means


# ------------------------------------------------------------------ table 2 --
def verbosity():
    QL = ["shortest", "short", "median", "long", "longest"]
    acc = {}
    tot = defaultdict(lambda: [0, 0])
    for r in rows("beh_verbosity.csv"):
        n, c = int(r["n"]), int(r["correct"])
        acc[(r["model"], r["family"], int(r["quintile"]))] = (n, c)
        if r["model"] in OPEN:
            tot[(r["family"], int(r["quintile"]))][0] += n
            tot[(r["family"], int(r["quintile"]))][1] += c
    body = [group("Open weights", 12)]
    for m in ORDER:
        if m == CLOSED[0]:
            body.append(group("API only", 12))
        cells = []
        for fam in ("single", "multi"):
            # One anchor per model per half. The multi-turn half sits lower
            # everywhere, so a shared anchor would paint it red end to end and
            # hide the length trend inside it -- which is the claim being made.
            mu, span = anchor([c / n for q in range(5)
                               for (n, c) in [acc.get((m, fam, q), (0, 0))] if n])
            for q in range(5):
                n, c = acc.get((m, fam, q), (0, 0))
                cells.append(ramp(c / n, mu, span) + num(c / n) if n else "---")
        a = acc.get((m, "multi", 0), (1, 0))
        z = acc.get((m, "multi", 4), (1, 0))
        drop = z[1] / max(1, z[0]) - a[1] / max(1, a[0])
        cells.append((cc("lvqaDown", min(72, round(abs(drop) / 0.35 * 72)))
                      if drop < 0 else cc("lvqaUp", min(58, round(drop / 0.15 * 58))))
                     + B + "nb{" + f"{drop:+.3f}".replace("0.", ".") + "}")
        body.append(DISPLAY[m] + " & " + " & ".join(cells) + B + B)
    body.append(B + "midrule")
    mean_cells = [num(tot[(f, q)][1] / tot[(f, q)][0])
                  for f in ("single", "multi") for q in range(5)]
    body.append(B + "textit{Mean, open} & " + " & ".join(mean_cells) + " & "
                + B + B)

    head = ["& " + B + "multicolumn{5}{c}{" + B + "textsc{evidence in one turn}} & "
            + B + "multicolumn{5}{c}{" + B + "textsc{evidence across turns}} & "
            + B + B,
            B + "cmidrule(lr){2-6}" + B + "cmidrule(lr){7-11}",
            "Model & " + " & ".join(QL) + " & " + " & ".join(QL)
            + " & $" + B + "Delta$ " + B + B]
    tex = PREAMBLE + wrap(body, "l ccccc ccccc c", head)
    (OUT / "tab_verbosity.tex").write_text(tex + "\n", encoding="utf-8")
    print("  tab_verbosity.tex")


# ------------------------------------------------------------------ table 3 --
def ungrounded():
    """Correct answers given before the supporting layer had been delivered."""
    # Only the conditions where the question can be asked: multi-turn, and
    # carrying a layer schedule. A single-turn condition delivers everything
    # before the model speaks, so its ungrounded count is zero by construction
    # and pooling it in silently deflates the column it lands in.
    SCHEDULED = {"B7": "Visual", "B8": "Visual", "B9": "Visual",
                 "B10": "Layered", "B11": "Layered",
                 "B12": "Order", "B13": "Order"}
    fam_acc = defaultdict(lambda: [0, 0, 0])
    for r in rows("beh_cond.csv"):
        fam = SCHEDULED.get(r["cond"])
        if fam is None:
            continue
        a = fam_acc[(r["model"], fam)]
        a[0] += int(r["n"])
        a[1] += int(r["correct"])
        # The conjunction, not the ungrounded count on its own. A model can be
        # right before its evidence arrives and then talk itself out of it, so
        # ungrounded is not a subset of correct and the bare ratio is not a
        # share -- it can exceed one.
        a[2] += int(r["both"])
    fams = ["Visual", "Layered", "Order"]
    # Table-wide here, unlike the other two: the absolute level is the finding,
    # so a model that is ungrounded everywhere should read that way.
    mu, span = anchor([fam_acc[(m, f)][2] / max(1, fam_acc[(m, f)][1])
                       for m in OPEN for f in fams])

    body = [group("Open weights", 6)]
    for m in ORDER:
        if m == CLOSED[0]:
            body.append(group("API only", 6))
        cells = []
        for f in fams:
            n, c, u = fam_acc[(m, f)]
            cells.append(ramp(u / max(1, c), mu, span, higher_is_better=False)
                         + num(u / max(1, c)))
        n = sum(fam_acc[(m, f)][0] for f in fams)
        c = sum(fam_acc[(m, f)][1] for f in fams)
        u = sum(fam_acc[(m, f)][2] for f in fams)
        cells.append(num(c / n))
        cells.append(B + "textbf{" + num((c - u) / n) + "}")
        body.append(DISPLAY[m] + " & " + " & ".join(cells) + B + B)
    body.append(B + "midrule")
    tot = [sum(fam_acc[(m, f)][i] for m in OPEN for f in fams) for i in range(3)]
    mean_cells = [num(sum(fam_acc[(m, f)][2] for m in OPEN)
                      / max(1, sum(fam_acc[(m, f)][1] for m in OPEN))) for f in fams]
    mean_cells += [num(tot[1] / tot[0]), B + "textbf{" + num((tot[1] - tot[2]) / tot[0]) + "}"]
    body.append(B + "textit{Mean, open} & " + " & ".join(mean_cells) + B + B)

    head = ["& " + B + "multicolumn{3}{c}{" + B
            + "textsc{ungrounded share}} & "
            + B + "multicolumn{2}{c}{" + B + "textsc{accuracy}} " + B + B,
            B + "cmidrule(lr){2-4}" + B + "cmidrule(lr){5-6}",
            "Model & Visual & Layered & Order & as scored & grounded only " + B + B]
    tex = PREAMBLE + wrap(body, "l ccc cc", head)
    (OUT / "tab_ungrounded.tex").write_text(tex + "\n", encoding="utf-8")
    print("  tab_ungrounded.tex")


first_attempt()
verbosity()
ungrounded()
