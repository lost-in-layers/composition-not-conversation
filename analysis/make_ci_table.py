"""Per-condition accuracy with a bootstrap interval, and the paired contrasts.

Questions are the unit of resampling, not conversations. The same 300 questions
appear in every condition and in every run, so a conversation-level interval
would treat three views of one question as three independent observations and
report an interval about twice too narrow.

Each contrast is computed within a bootstrap replicate, so the interval is on
the difference rather than on two separately resampled means. That is what makes
"negative for all eleven models" and "the interval excludes zero" say the same
thing.

Usage: python overleaf/make_ci_table.py <csv-dir>
"""
import csv
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

SRC = Path(sys.argv[1])
OUT = Path(__file__).parent
B = chr(92)
BOOT = 2000
SEED = 0

NAMES = [("B1", "full"), ("B2", "concat-q"), ("B3", "sharded-q"),
         ("B4", "recap-q"), ("B5", "snowball-q"), ("B6", "concat-v"),
         ("B7", "sharded-v"), ("B8", "recap-v"), ("B9", "snowball-v"),
         ("B10", "aligned"), ("B11", "misaligned"), ("B12", "evidence-first"),
         ("B13", "evidence-last"), ("B14", "shuffled"),
         ("A2", "all-layers"), ("A3", "oracle")]
CONTRASTS = [("B1", "B2", "question in parts, one turn"),
             ("B2", "B3", "question spread over turns"),
             ("B1", "B6", "scene in parts, one turn"),
             ("B6", "B7", "scene spread over turns"),
             ("B7", "B8", "scene restored at the end"),
             ("B1", "B8", "restored scene against the baseline"),
             ("B1", "A3", "oracle against the whole scene"),
             ("B1", "A2", "all layers beside the whole scene")]

# task -> cond -> [runs, correct]
per = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for row in csv.DictReader((SRC / "beh_question.csv").open(encoding="utf-8")):
    a = per[row["task_id"]][row["cond"]]
    a[0] += int(row["runs"])
    a[1] += int(row["correct"])
tasks = sorted(per)


def mean_over(sample, cond):
    n = sum(per[t][cond][0] for t in sample)
    x = sum(per[t][cond][1] for t in sample)
    return x / n if n else float("nan")


# Resample scenes, not questions. The 300 questions come from 93 scenes, and
# questions on one scene share an image, so they are not independent. Drawing a
# scene brings all of its questions with it.
scenes = defaultdict(list)
for t in tasks:
    scenes[t.split("::")[0]].append(t)
scene_ids = sorted(scenes)
rng = random.Random(SEED)
draws = []
for _ in range(BOOT):
    sample = []
    for _ in scene_ids:
        sample += scenes[scene_ids[rng.randrange(len(scene_ids))]]
    draws.append(sample)


def interval(values):
    values = sorted(values)
    lo = values[int(0.025 * len(values))]
    hi = values[int(0.975 * len(values)) - 1]
    return lo, hi


def num(x):
    return f"{x:.3f}".lstrip("0")


def signed(x):
    return f"{x:+.3f}".replace("0.", ".")


L = [B + "begin{tabular}{@{}llcc@{}}", B + "toprule",
     "Condition & & Accuracy & 95" + B + "% interval " + B + B, B + "midrule"]
for key, label in NAMES:
    point = mean_over(tasks, key)
    lo, hi = interval([mean_over(d, key) for d in draws])
    L.append(B + "textsc{" + label + "} & & " + num(point)
             + " & [" + num(lo) + ", " + num(hi) + "] " + B + B)
L += [B + "midrule",
      B + "multicolumn{4}{@{}l}{" + B + "textit{Paired contrasts}} " + B + B]
for a, b, label in CONTRASTS:
    point = mean_over(tasks, b) - mean_over(tasks, a)
    lo, hi = interval([mean_over(d, b) - mean_over(d, a) for d in draws])
    L.append(label + " & " + B + "textsc{" + dict(NAMES)[b] + "} "
             + B + "textbf{--} " + B + "textsc{" + dict(NAMES)[a] + "} & "
             + signed(point) + " & [" + signed(lo) + ", " + signed(hi) + "] "
             + B + B)
L += [B + "bottomrule", B + "end{tabular}"]

(OUT / "tab_ci.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"wrote tab_ci.tex  ({BOOT} draws over {len(scene_ids)} scenes, {len(tasks)} questions)")
for a, b, label in CONTRASTS:
    point = mean_over(tasks, b) - mean_over(tasks, a)
    lo, hi = interval([mean_over(d, b) - mean_over(d, a) for d in draws])
    print(f"  {label:38} {point:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
