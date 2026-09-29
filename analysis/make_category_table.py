"""Accuracy by question category, for the appendix.

This is the table that turns the two headline losses into one mechanism. Both
the oracle inversion and the compositional loss concentrate on the questions
that ask about a relation between objects, and both are small on the questions
that ask about one object on its own. Fragmenting an image keeps the objects
and destroys what holds between them.

Usage: python overleaf/make_category_table.py <csv-dir> <lic-dir>
"""
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

SRC = Path(sys.argv[1])
LIC = Path(sys.argv[2] if len(sys.argv) > 2 else "data/derived/lic")
OUT = Path(__file__).parent
B = chr(92)

CATS = [("direct_ask", "Direct ask"), ("attribute_binding", "Attribute binding"),
        ("occlusion", "Occlusion"), ("counting", "Counting"),
        ("spatial", "Spatial"), ("multi_hop_reference", "Multi-hop")]
RELATIONAL = {"counting", "spatial", "multi_hop_reference"}
COLS = [("B1", "full"), ("B2", "concat-q"), ("B3", "sharded-q"),
        ("B6", "concat-v"), ("B7", "sharded-v"), ("B8", "recap-v"),
        ("A2", "all-lyr"), ("A3", "oracle")]

cat, nq = {}, defaultdict(int)
data = json.loads((LIC / "layered_vqa_oracle_gold.json").read_text(encoding="utf-8"))
for r in (data if isinstance(data, list) else data.values()):
    cat[r["task_id"]] = r["meta"]["category"]
    nq[r["meta"]["category"]] += 1

agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for row in csv.DictReader((SRC / "beh_question.csv").open(encoding="utf-8")):
    c = cat.get(row["task_id"])
    if c is None:
        continue
    a = agg[row["cond"]][c]
    a[0] += int(row["runs"])
    a[1] += int(row["correct"])


def acc(cond, group):
    n = sum(agg[cond][c][0] for c in group)
    x = sum(agg[cond][c][1] for c in group)
    return x / n if n else float("nan")


def num(x):
    return f"{x:.3f}".lstrip("0")


def signed(x):
    return f"{x:+.3f}".replace("0.", ".")


def cc(colour, pct):
    return B + "cellcolor{" + colour + "!" + str(int(pct)) + "}"


def shade(delta, cap=0.25):
    """Red below the model's own FULL score, teal above, capped at a 25 point move."""
    t = max(-1.0, min(1.0, delta / cap))
    return cc("lvqaDown", round(60 * -t)) if t < 0 else cc("lvqaUp", round(52 * t))


L = [B + "begin{tabular}{l r *{7}{c}}", B + "toprule"]
L.append("Question type & $n$ & " + B + "textsc{full} & "
         + " & ".join(B + "textsc{" + lab + "}" for _k, lab in COLS[1:])
         + " " + B + B)
L.append(B + "midrule")
for key, label in CATS:
    base = acc("B1", [key])
    cells = [B + "cellcolor{lvqaBase!30}" + B + "textbf{" + num(base) + "}"]
    cells += [shade(acc(k, [key]) - base) + num(acc(k, [key])) for k, _l in COLS[1:]]
    L.append(f"{label} & {nq[key]} & " + " & ".join(cells) + " " + B + B)
L.append(B + "midrule")
for label, group in (("Single-object", [c for c, _ in CATS if c not in RELATIONAL]),
                     ("Relational", [c for c, _ in CATS if c in RELATIONAL])):
    base = acc("B1", group)
    n = sum(nq[c] for c in group)
    cells = [B + "cellcolor{lvqaBase!30}" + B + "textbf{" + num(base) + "}"]
    cells += [shade(acc(k, group) - base) + signed(acc(k, group) - base)
              for k, _l in COLS[1:]]
    L.append(B + "textit{" + label + "} & " + str(n) + " & "
             + " & ".join(cells) + " " + B + B)
L += [B + "bottomrule", B + "end{tabular}"]

(OUT / "tab_category.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
print("wrote tab_category.tex")
for label, group in (("single-object", [c for c, _ in CATS if c not in RELATIONAL]),
                     ("relational", [c for c, _ in CATS if c in RELATIONAL])):
    b = acc("B1", group)
    print(f"  {label:14} oracle {acc('A3', group) - b:+.3f}   "
          f"layers {acc('B6', group) - b:+.3f}   "
          f"sharded-v {acc('B7', group) - b:+.3f}")
