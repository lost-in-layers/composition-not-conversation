"""Publication figures for the open-model study, as PDF and PNG.

Every figure here answers the same question one of the Lost-in-Conversation
figures asks, but none of them is that figure. Where their design throws
information away, this one keeps it:

  reliability   they need three explainer panels plus a 15x3 boxplot grid to
                separate aptitude from unreliability, and their unreliability
                is a spread over repeated simulations. We have three exact
                replicates of every question, so unreliability can be measured
                directly -- the share of questions the model answers
                inconsistently across runs -- and both quantities fit in one
                plane with an arrow per model.
  shard count   theirs is a summary for two models; ours covers eleven and
                separates text shards from layer shards, which is the split
                the benchmark exists to test.
  answer bloat  theirs plots length alone. Length is only interesting if it
                predicts failure, so accuracy of the attempt is plotted on the
                same axis.
  citations     theirs is one triangle of "which turn's document is cited".
                Ours is two -- gold layers and distractor layers -- because the
                interesting failure is citing evidence that does not support
                the question, which a single triangle cannot show.

Usage:  python overleaf/make_figures.py <csv-dir> <out-dir>
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

DOWN = "#CA4A38"
UP = "#2F7C8C"
GREY = "#787878"
INK = "#1a1a1a"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 7.5,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.6,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})

ORDER = ["qwen2.5-vl-3b", "qwen3-vl-4b", "qwen2.5-vl-7b", "qwen3-vl-8b",
         "internvl3.5-8b", "ministral3-8b-instruct", "qwen3.5-9b", "pixtral-12b",
         "qwen3.6-27b", "qwen3-vl-30b-a3b", "qwen2.5-vl-32b"]
SHORT = {"qwen2.5-vl-3b": "Q2.5-VL-3B", "qwen3-vl-4b": "Q3-VL-4B",
         "qwen2.5-vl-7b": "Q2.5-VL-7B", "qwen3-vl-8b": "Q3-VL-8B",
         "internvl3.5-8b": "IVL3.5-8B", "ministral3-8b-instruct": "Min-3-8B",
         "qwen3.5-9b": "Q3.5-9B", "pixtral-12b": "Pixtral-12B",
         "qwen3.6-27b": "Q3.6-27B", "qwen3-vl-30b-a3b": "Q3-VL-30B",
         "qwen2.5-vl-32b": "Q2.5-VL-32B"}

CMAP = LinearSegmentedColormap.from_list("lvqa", [DOWN, "#f2efe9", UP])
SEQ = LinearSegmentedColormap.from_list("lvqaseq", ["#ffffff", UP, "#123c45"])


def rows(name):
    return list(csv.DictReader((SRC / name).open(encoding="utf-8")))


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  {stem}.pdf / .png")


def place_labels(ax, points, fontsize=5.4):
    """Label each point, trying candidate offsets until one does not collide.

    Eleven arrows in one plane put several heads within a few points of each
    other, and a fixed offset then stacks the names on top of one another. The
    candidates are ordered by preference, so most labels still land below-left
    and the layout stays regular; only the crowded ones move.
    """
    fig = ax.figure
    fig.canvas.draw()
    placed = []
    candidates = [(-7, -9), (7, -9), (-7, 6), (7, 6), (0, -14), (0, 11),
                  (-14, -2), (14, -2), (-7, -17), (7, -17), (-7, 14), (7, 14),
                  (0, -21), (0, 18)]
    for x, y, text in points:
        px, py = ax.transData.transform((x, y))
        chosen = candidates[0]
        for dx, dy in candidates:
            cx, cy = px + dx, py + dy
            half_w = 2.1 * len(text) + 3
            if not any(abs(cx - ox) < half_w + owh and abs(cy - oy) < 6.0
                       for ox, oy, owh in placed):
                chosen = (dx, dy)
                break
        cx, cy = px + chosen[0], py + chosen[1]
        placed.append((cx, cy, 2.1 * len(text) + 3))
        ax.annotate(text, (x, y), textcoords="offset points", xytext=chosen,
                    ha="right" if chosen[0] < 0 else ("left" if chosen[0] > 0 else "center"),
                    fontsize=fontsize, color=INK,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                              alpha=0.8))


# ---------------------------------------------------------------- figure 1 --
def fig_reliability(stem="fig_reliability", titled=True):
    """Aptitude against unreliability, with an arrow per model.

    Aptitude is accuracy. Unreliability is the share of questions whose verdict
    is not unanimous across the three replicate runs -- the model got it right
    once and wrong another time, on the same question, under identical
    settings. That is a property of the model, not of our sample size, which is
    what makes it worth plotting against aptitude.
    """
    agg = defaultdict(lambda: [0, 0])
    for r in rows("beh_question.csv"):
        runs, correct = int(r["runs"]), int(r["correct"])
        if runs < 3:
            continue
        a = agg[(r["model"], r["cond"])]
        a[0] += 1
        a[1] += 1 if 0 < correct < runs else 0
    acc = {}
    for r in rows("beh_cond.csv"):
        acc[(r["model"], r["cond"])] = int(r["correct"]) / max(1, int(r["n"]))

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0), sharey=True)
    pairs = [("B1", "B3", "Text: full question $\\rightarrow$ sharded"),
             ("B6", "B7", "Layers: all at once $\\rightarrow$ sharded")]
    for ax, (a_key, b_key, title) in zip(axes, pairs):
        heads = []
        for m in ORDER:
            x0, x1 = acc[(m, a_key)], acc[(m, b_key)]
            y0 = agg[(m, a_key)][1] / agg[(m, a_key)][0]
            y1 = agg[(m, b_key)][1] / agg[(m, b_key)][0]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.7,
                                        alpha=0.75, shrinkA=2.5, shrinkB=3))
            ax.scatter([x0], [y0], s=17, facecolor="white", edgecolor=INK,
                       linewidth=0.8, zorder=3)
            ax.scatter([x1], [y1], s=22, facecolor=DOWN, edgecolor="white",
                       linewidth=0.6, zorder=3)
            heads.append((x1, y1, SHORT[m]))
        place_labels(ax, heads)
        ax.set_title(title)
        ax.set_xlabel("aptitude  (accuracy)")
        ax.grid(alpha=0.18, linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("unreliability\n(share of questions decided both ways)")
    axes[0].scatter([], [], s=17, facecolor="white", edgecolor=INK,
                    linewidth=0.8, label="evidence in one turn")
    axes[0].scatter([], [], s=22, facecolor=DOWN, edgecolor="white", label="evidence sharded")
    # Legend below the axes: inside, it competes with the labels for the same
    # empty corner and the placement search then has nowhere left to go.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.07), handletextpad=0.4)
    if titled:
        fig.suptitle("Sharding costs aptitude and buys unreliability",
                     y=1.02, fontsize=9)
    save(fig, stem)


def fig_reliability_plain():
    """The same figure with no suptitle, for a caption to carry instead."""
    fig_reliability(stem="fig_reliability_plain", titled=False)


# ---------------------------------------------------------------- figure 2 --
def fig_shards():
    """Accuracy against the number of turns the evidence was split across."""
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows("beh_shards.csv"):
        s = int(r["shards"])
        if not 2 <= s <= 8:
            continue
        a = agg[r["family"]][(r["model"], s)]
        a[0] += int(r["n"])
        a[1] += int(r["correct"])
    fams = [f for f in ("Text", "Visual", "Layered", "Order") if agg[f]]
    fig, axes = plt.subplots(1, len(fams), figsize=(2.05 * len(fams), 2.5),
                             sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, fams):
        grand = defaultdict(lambda: [0, 0])
        for m in ORDER:
            xs, ys = [], []
            for s in range(2, 9):
                n, c = agg[fam][(m, s)]
                if n >= 40:
                    xs.append(s)
                    ys.append(c / n)
                    grand[s][0] += n
                    grand[s][1] += c
            if xs:
                ax.plot(xs, ys, color=GREY, lw=0.6, alpha=0.45, zorder=1)
        gx = sorted(grand)
        ax.plot(gx, [grand[s][1] / grand[s][0] for s in gx], color=DOWN, lw=1.9,
                marker="o", ms=3.4, zorder=3, label="mean")
        ax.set_title(fam)
        ax.set_xlabel("shards in conversation")
        ax.grid(alpha=0.18, linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Accuracy falls with every extra turn the evidence is split across",
                 y=1.03, fontsize=9)
    save(fig, "fig_shards")


# ---------------------------------------------------------------- figure 3 --
def fig_bloat(stem="fig_bloat", titled=True):
    """Response length by attempt number, with the accuracy of those attempts.

    This is where our benchmark parts company with the text-only result. There,
    answers grow as the conversation drags on. Here they shrink: the first
    attempt is the longest because the model is still negotiating -- asking what
    the target object is, describing what it sees -- and by the last attempt it
    emits a bare token. Length alone would leave that ambiguous, so the accuracy
    of the very same attempts is drawn on the second axis, and it is the more
    useful curve: accuracy peaks mid-conversation and falls away after.
    """
    agg = defaultdict(lambda: [0, 0.0, 0])
    ref = defaultdict(lambda: [0, 0.0])
    for r in rows("beh_bloat.csv"):
        key = (r["family"], int(r["attempt"]))
        n, ch, co = int(r["n"]), float(r["chars"]), int(r["correct"])
        if int(r["single_turn"]):
            ref[r["family"]][0] += n
            ref[r["family"]][1] += ch
            continue
        a = agg[key]
        a[0] += n
        a[1] += ch
        a[2] += co
    fams = ["Text", "Visual", "Layered"]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5), sharex=True)
    for ax, fam in zip(axes, fams):
        xs = [i for i in range(1, 11) if agg[(fam, i)][0] >= 60]
        if not xs:
            continue
        length = [agg[(fam, i)][1] / agg[(fam, i)][0] for i in xs]
        ax.plot(xs, length, color=DOWN, lw=1.7, marker="o", ms=3.4, zorder=3)
        if ref[fam][0]:
            y = ref[fam][1] / ref[fam][0]
            ax.axhline(y, color=GREY, lw=0.9, ls=(0, (4, 2)))
            ax.annotate(f"one-turn: {y:.0f}", (xs[-1], y), fontsize=6, color=GREY,
                        va="bottom", ha="right")
        ax.set_title(fam)
        ax.set_xlabel("answer attempt")
        ax.grid(alpha=0.18, linewidth=0.5)
        ax.set_axisbelow(True)
        twin = ax.twinx()
        twin.plot(xs, [agg[(fam, i)][2] / agg[(fam, i)][0] for i in xs],
                  color=UP, lw=1.2, ls=(0, (2, 1.6)), marker="s", ms=2.6)
        twin.set_ylim(0, 1)
        twin.tick_params(axis="y", colors=UP, labelsize=6)
        if ax is axes[-1]:
            twin.set_ylabel("accuracy of that attempt", color=UP, fontsize=7)
        else:
            twin.set_yticklabels([])
    axes[0].set_ylabel("response length (chars)", color=DOWN)
    axes[0].tick_params(axis="y", colors=DOWN)
    if titled:
        fig.suptitle("Attempts get terser, not longer; accuracy peaks mid-conversation",
                     y=1.03, fontsize=9)
    save(fig, stem)


def fig_bloat_plain():
    """The same figure with no suptitle, for a caption to carry instead."""
    fig_bloat(stem="fig_bloat_plain", titled=False)


# ---------------------------------------------------------------- figure 4 --
def fig_citation(stem="fig_citation", titled=True):
    """When a layer is introduced vs when it is still being cited.

    Two triangles, because citing a gold layer late is recovery and citing a
    distractor late is the failure the benchmark was built to catch. Row-
    normalised: each row is one introduction turn.
    """
    mats = {}
    for role in ("gold", "distractor"):
        M = np.zeros((7, 7))
        for r in rows("beh_citation.csv"):
            if r["role"] != role:
                continue
            i, j = int(r["intro_turn"]), int(r["mention_turn"])
            if 1 <= i <= 7 and 1 <= j <= 7:
                M[i - 1, j - 1] += int(r["count"])
        mats[role] = M
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9))
    for ax, role in zip(axes, ("gold", "distractor")):
        M = mats[role]
        tot = M.sum(axis=1, keepdims=True)
        P = np.divide(M, np.where(tot == 0, 1, tot))
        # A layer cannot be cited before it is introduced, so the strictly lower
        # triangle is not empty data -- it is impossible data, and is masked out.
        masked = np.ma.masked_where(np.tril(np.ones_like(P), -1).astype(bool), P)
        im = ax.imshow(masked, cmap=SEQ, vmin=0, vmax=0.5, aspect="auto")
        for i in range(7):
            for j in range(7):
                if j >= i and tot[i, 0] > 0:
                    ax.text(j, i, f"{P[i, j]*100:.0f}", ha="center", va="center",
                            fontsize=6, color="white" if P[i, j] > 0.30 else INK)
        ax.set_xticks(range(7), [str(i + 1) for i in range(7)])
        ax.set_yticks(range(7), [str(i + 1) for i in range(7)])
        ax.set_xlabel("turn the layer is cited in")
        ax.set_title(f"{role} layers", color=UP if role == "gold" else DOWN)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03).ax.tick_params(labelsize=6)
    axes[0].set_ylabel("turn the layer was introduced")
    if titled:
        fig.suptitle("Where each layer is cited, relative to the turn it arrived",
                     y=1.02, fontsize=9)
    save(fig, stem)


def fig_citation_plain():
    """The same figure with no suptitle, for a caption to carry instead."""
    fig_citation(stem="fig_citation_plain", titled=False)


# ---------------------------------------------------------------- figure 5 --
def fig_judges(stem="fig_judges", figsize=(5.6, 3.2), annot=5.2,
               ticks=6, titled=True):
    """Cross-judge matrix with the self-preference residual on the diagonal.

    ability + leniency - grand mean is what a cell would read if judges had no
    preferences; the residual is what is left over. The diagonal is the only
    place a model grades itself, so a positive diagonal residual is
    self-preference and nothing else.
    """
    cell, jn, an = {}, defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for r in rows("beh_judge.csv"):
        n, c = int(r["n"]), int(r["correct"])
        cell[(r["judge"], r["assistant"])] = c / n
        jn[r["judge"]][0] += n
        jn[r["judge"]][1] += c
        an[r["assistant"]][0] += n
        an[r["assistant"]][1] += c
    judges = [m for m in ORDER if m in jn]
    grand = sum(jn[j][1] for j in judges) / sum(jn[j][0] for j in judges)
    R = np.full((len(judges), len(ORDER)), np.nan)
    for i, j in enumerate(judges):
        lenience = jn[j][1] / jn[j][0]
        for k, a in enumerate(ORDER):
            if (j, a) in cell:
                R[i, k] = cell[(j, a)] - (an[a][1] / an[a][0] + lenience - grand)
    fig, ax = plt.subplots(figsize=figsize)
    lim = float(np.nanmax(np.abs(R)))
    im = ax.imshow(R, cmap=CMAP, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(judges)):
        for k in range(len(ORDER)):
            if not np.isnan(R[i, k]):
                ax.text(k, i, f"{R[i, k]*100:+.0f}", ha="center", va="center",
                        fontsize=annot,
                        color="white" if abs(R[i, k]) > lim * 0.62 else INK)
        if judges[i] in ORDER:
            k = ORDER.index(judges[i])
            ax.add_patch(plt.Rectangle((k - .5, i - .5), 1, 1, fill=False,
                                       edgecolor=INK, lw=1.3, zorder=5))
    ax.set_xticks(range(len(ORDER)), [SHORT[m] for m in ORDER], rotation=60,
                  ha="right", fontsize=ticks)
    ax.set_yticks(range(len(judges)), [SHORT[m] for m in judges],
                  fontsize=ticks)
    ax.set_xlabel("model being graded")
    ax.set_ylabel("model doing the grading")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                 label="residual (pts)").ax.tick_params(labelsize=ticks)
    if titled:
        ax.set_title("What is left after ability and leniency are removed\n"
                     "(boxed = model grading itself)", fontsize=8.5)
    save(fig, stem)


def fig_judges_wrap():
    """The same matrix, sized for a wrapfigure.

    A wrapped float is scaled to a fraction of the column, so every font inside
    it shrinks with the box. The cell numbers were set for a full-width float
    and stop being readable at that size. This variant enlarges them and drops
    the title, which the LaTeX caption carries anyway.
    """
    fig_judges(stem="fig_judges_wrap", figsize=(6.2, 3.6), annot=7.6,
               ticks=8, titled=False)


for f in (fig_reliability, fig_reliability_plain, fig_shards,
          fig_bloat, fig_bloat_plain, fig_citation, fig_citation_plain,
          fig_judges, fig_judges_wrap):
    f()
print("figures written to", OUT)
