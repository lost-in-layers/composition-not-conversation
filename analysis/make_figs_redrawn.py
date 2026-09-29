#!/usr/bin/env python3
"""Produce a restrained revision of the original paper figures.

The original visual language is preserved. This script only rebuilds Figure 1
as a shorter horizontal teaser, emits Figures 2--4 as independently placeable
subfigures, and writes everything to ``overleaf/figures_redrawn``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

import make_figs as D


HERE = Path(__file__).resolve().parent
OUT = HERE / "figures_redrawn"
OUT.mkdir(exist_ok=True)

MT = D.MT
MODELS, MODEL_IDS = D.MODELS, D.MODEL_IDS
C, T, MC, MDISP = D.C, D.T, D.MC, D.MDISP
W = D.W


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", facecolor="white")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def tidy(ax, grid="y") -> None:
    D.tidy(ax, grid)


def short_signed(x: float) -> str:
    return ("+" if x >= 0 else "−") + f"{abs(x) * 100:.0f}"


# ---------------------------------------------------------------------------
# Figure 1 -- same three-claim story, compressed into a horizontal teaser
# ---------------------------------------------------------------------------


def fig1_teaser() -> None:
    fig = plt.figure(figsize=(W, 3.05))
    fig.text(0.5, 0.970, "Vision–language models lose the scene, not the thread",
             ha="center", va="top", fontsize=10.2, color=C["ink"],
             fontweight="bold")
    fig.text(0.5, 0.925,
             "The same source image, reorganized three ways — and what each reorganization reveals",
             ha="center", va="top", fontsize=6.4, color=C["mute"], style="italic")

    cols = [(0.018, 0.318), (0.350, 0.650), (0.682, 0.982)]
    heads = [
        ("Loss in composition", "all the parts, without the scene", C["parts"], T["parts"]),
        ("Oracle inversion", "only the annotated evidence", C["oracle"], T["oracle"]),
        ("Loss in grounding", "the right answer, the wrong support", C["evid"], T["evid"]),
    ]
    for (x0, x1), (title, sub, col, wash) in zip(cols, heads):
        D.zone(fig, x0, 0.018, x1, 0.875, wash, r=0.015)
        fig.text((x0 + x1) / 2, 0.842, title, ha="center", va="top",
                 fontsize=7.8, color=col, fontweight="bold")
        fig.text((x0 + x1) / 2, 0.805, sub, ha="center", va="top",
                 fontsize=5.7, color=C["mute"], style="italic")

    hero = D.HERO
    ghero = D.HERO_G
    meta = D.CANON[hero]["meta"]
    all_ids = sorted(int(k) for k in meta["layer_labels"])
    gold = meta["gold_layer_ids"]

    def image_panel(rect, image, frame, title, ok):
        ax = fig.add_axes(rect)
        D.show(ax, image, frame=frame, lw=1.05)
        ax.set_title(title, fontsize=5.6, color=frame, pad=1.7)
        D.badge(ax, 0.91, 0.13, ok=ok, r=0.060, lw=1.05)
        return ax

    # Compact paired image rows: no vertical stacks and no overflowing cards.
    image_panel((0.035, 0.555, 0.125, 0.180),
                D._fit(Image.open(D.composite(hero)), (360, 235)),
                C["whole"], "whole", True)
    image_panel((0.177, 0.555, 0.125, 0.180),
                D.montage(D.layer_files(hero, [0] + all_ids), cols=3, cell=(120, 78)),
                C["parts"], "layers", False)
    fig.text(0.168, 0.640, "→", color=C["parts"], fontsize=10,
             ha="center", va="center", fontweight="bold")

    image_panel((0.367, 0.555, 0.125, 0.180),
                D._fit(Image.open(D.composite(hero)), (360, 235)),
                C["whole"], "whole", True)
    image_panel((0.509, 0.555, 0.125, 0.180),
                D.montage(D.layer_files(hero, gold), cols=2, cell=(150, 100),
                          highlight=set(range(len(gold)))),
                C["oracle"], "gold only", False)
    fig.text(0.500, 0.640, "→", color=C["oracle"], fontsize=10,
             ha="center", va="center", fontweight="bold")

    image_panel((0.699, 0.555, 0.125, 0.180),
                D._fit(Image.open(D.composite(ghero)), (360, 235)),
                C["whole"], "answer correct", True)
    image_panel((0.841, 0.555, 0.125, 0.180),
                D.montage(D.layer_files(ghero, D.HERO_G_GOLD + D.HERO_G_PRED),
                          cols=2, cell=(150, 100), highlight={1}),
                C["evid"], "gold | cited", False)
    fig.text(0.832, 0.640, "→", color=C["evid"], fontsize=10,
             ha="center", va="center", fontweight="bold")

    captions = [
        "same visual content;  −17 points",
        "minimal evidence;  all 6 models worsen",
        "answer: Blue;  cited: zebra",
    ]
    for (x0, x1), text, col in zip(cols, captions,
                                    [C["parts"], C["oracle"], C["evid"]]):
        fig.text((x0 + x1) / 2, 0.505, text, ha="center", va="center",
                 fontsize=5.7, color=col, fontweight="bold")

    # Bottom-left: the causal composition ladder.
    ax = fig.add_axes((0.060, 0.145, 0.227, 0.290))
    keys = ["B1", "B6", "B7", "B8"]
    labels = ["whole", "parts\n1 turn", "parts\nturns", "recap"]
    x = np.arange(4)
    for m, _ in MODELS:
        ax.plot(x, [MT.acc(m, k) for k in keys], "-o", color=MC[m], lw=0.95,
                ms=1.9, markeredgewidth=0, alpha=0.70, zorder=1)
    mean = [D.macro(k) for k in keys]
    ax.plot(x, mean, "-o", color=C["ink"], lw=2.1, ms=3.8, zorder=5,
            markeredgecolor="white", markeredgewidth=0.55)
    D.drop_arrow(ax, 0.07, mean[0] - 0.005, 0.92, mean[1] + 0.005,
                 C["parts"], lw=1.05)
    D.drop_arrow(ax, 2.08, mean[2] + 0.005, 2.92, mean[3] - 0.005,
                 C["recap"], lw=1.05)
    ax.text(0.26, 0.90, "−17 pts", transform=ax.transAxes, color=C["parts"],
            fontsize=6.1, fontweight="bold", ha="center")
    ax.text(0.78, 0.90, "+10 pts", transform=ax.transAxes, color=C["recap"],
            fontsize=6.1, fontweight="bold", ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.1)
    ax.set_ylim(0.33, 0.78)
    ax.set_ylabel("answer acc.", fontsize=5.8, labelpad=1.0)
    ax.yaxis.set_label_coords(-0.13, 0.5)
    tidy(ax)

    # Bottom-middle: the oracle penalty by category.
    ax = fig.add_axes((0.420, 0.145, 0.205, 0.290))
    rows = []
    ids = sorted(MT.DATASET_IDS["layered_vqa_text.json"])
    for c in D.CATS:
        tids = [t for t in ids if D.cat(t) == c]
        rows.append((c, D.acc_over("A1", tids), D.acc_over("A3", tids)))
    rows.sort(key=lambda z: z[2] - z[1], reverse=True)
    for i, (cat, full, oracle) in enumerate(rows):
        col = C["oracle"] if cat in D.RELATIONAL else C["mute"]
        D.drop_arrow(ax, full, i, oracle, i, col, lw=1.3)
        ax.scatter(full, i, s=19, color=C["whole"], edgecolor="white", lw=0.45, zorder=4)
        ax.scatter(oracle, i, s=19, color=C["oracle"], edgecolor="white", lw=0.45, zorder=4)
    ax.axhline(1.5, color=C["mute"], lw=0.65, alpha=0.65, zorder=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([D.CAT_SHORT[c] for c, _, _ in rows], fontsize=4.9)
    ax.invert_yaxis()
    ax.set_xlim(0.30, 0.82)
    ax.set_xlabel("answer accuracy", fontsize=5.8, labelpad=1.0)
    tidy(ax, "x")
    ax.text(0.805, 0.5, "needs the\nobject",
            color=C["mute"], fontsize=4.7, ha="right", va="center",
            linespacing=1.05)
    ax.text(0.805, 4.45, "needs the\nscene",
            color=C["oracle"], fontsize=4.7, fontweight="bold",
            ha="right", va="center", linespacing=1.05)

    # Bottom-right: grounding separates from answering.
    ax = fig.add_axes((0.730, 0.145, 0.223, 0.290))
    levels = sorted({D.demand(t) for t in D.GRND})
    answer, evidence = [], []
    for level in levels:
        sel = [g for g in D.GR if D.demand(g["task_id"]) == level]
        answer.append(np.mean([g["correct"] for g in sel]))
        evidence.append(np.mean([g["em"] for g in sel]))
    ax.fill_between(levels, evidence, answer, color=C["evid"], alpha=0.13, zorder=0)
    ax.plot(levels, answer, "-o", color=C["whole"], lw=1.7, ms=3.6,
            markeredgecolor="white", markeredgewidth=0.5)
    ax.plot(levels, evidence, "-o", color=C["evid"], lw=1.7, ms=3.6,
            markeredgecolor="white", markeredgewidth=0.5)
    ax.text(levels[0] + 0.08, answer[0] + 0.030, "answer", color=C["whole"],
            fontsize=5.5, fontweight="bold")
    ax.text(levels[0] + 0.08, evidence[0] - 0.080, "evidence", color=C["evid"],
            fontsize=5.5, fontweight="bold")
    gap_x = levels[-1] - 0.10
    ax.annotate("", xy=(gap_x, evidence[-1] + 0.015),
                xytext=(gap_x, answer[-1] - 0.015),
                arrowprops=dict(arrowstyle="<->", color=C["evid"], lw=1.0),
                zorder=6)
    ax.text(gap_x - 0.08, (answer[-1] + evidence[-1]) / 2,
            "grounding\ngap", color=C["evid"], fontsize=5.1,
            ha="right", va="center", linespacing=1.05)
    ax.set_xticks(levels)
    ax.set_xlabel("evidence demand", fontsize=5.8, labelpad=1.0)
    ax.set_ylim(-0.02, 0.77)
    ax.set_ylabel("accuracy", fontsize=5.8, labelpad=1.0)
    ax.yaxis.set_label_coords(-0.13, 0.5)
    tidy(ax)

    save(fig, "fig1_teaser")


# ---------------------------------------------------------------------------
# Figure 2 -- original aesthetic, independent panels
# ---------------------------------------------------------------------------


def oracle_rows():
    ids = sorted(MT.DATASET_IDS["layered_vqa_text.json"])
    rows = []
    for c in D.CATS:
        tids = [t for t in ids if D.cat(t) == c]
        a1, a3 = D.acc_over("A1", tids), D.acc_over("A3", tids)
        rows.append((c, a1, a3, len(tids), a3 - a1))
    return sorted(rows, key=lambda r: r[4], reverse=True)


def fig2_oracle() -> None:
    rows = oracle_rows()

    fig, ax = plt.subplots(figsize=(3.10, 2.18))
    fig.subplots_adjust(left=0.35, right=0.95, top=0.78, bottom=0.18)
    rel_start = next(i for i, r in enumerate(rows) if r[0] in D.RELATIONAL)
    ax.axhspan(rel_start - 0.5, len(rows) - 0.5, color=T["oracle"], zorder=0)
    for i, (cat, a1, a3, n, delta) in enumerate(rows):
        col = C["oracle"] if cat in D.RELATIONAL else C["mute"]
        D.drop_arrow(ax, a1, i, a3, i, col, lw=1.25)
        ax.scatter(a1, i, s=29, color=C["whole"], edgecolor="white", lw=0.65, zorder=5)
        ax.scatter(a3, i, s=29, color=C["oracle"], edgecolor="white", lw=0.65, zorder=5)
        ax.text(0.815, i, D.signed(delta), fontsize=6.3, color=col,
                fontweight="bold" if cat in D.RELATIONAL else "normal",
                ha="right", va="center")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{D.CAT_DISP[c]}  ({n})" for c, _, _, n, _ in rows], fontsize=6.1)
    ax.invert_yaxis()
    ax.set_xlim(0.30, 0.82)
    ax.set_xlabel("answer accuracy")
    ax.set_title("Removing the scene hurts relations most", fontsize=7.3,
                 loc="left", pad=16)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=C["whole"], ms=4, label="whole image"),
        Line2D([], [], marker="o", ls="", color=C["oracle"], ms=4, label="gold evidence only"),
    ], loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=2,
       handletextpad=0.25, columnspacing=0.8, borderaxespad=0.0, handlelength=0.8)
    ax.text(0.312, 0.5, "questions about objects", fontsize=5.6,
            color=C["mute"], ha="left", va="center")
    ax.text(0.312, rel_start + 0.5, "questions about arrangements", fontsize=5.6,
            color=C["oracle"], ha="left", va="center", fontweight="bold")
    tidy(ax, "x")
    save(fig, "fig2-a")

    fig, ax = plt.subplots(figsize=(2.52, 2.18))
    fig.subplots_adjust(left=0.39, right=0.94, top=0.78, bottom=0.18)
    order = sorted(MODEL_IDS, key=lambda m: MT.acc(m, "A3") - MT.acc(m, "A1"))
    for i, m in enumerate(order):
        a1, a3 = MT.acc(m, "A1"), MT.acc(m, "A3")
        D.drop_arrow(ax, a1, i, a3, i, C["oracle"], lw=1.25)
        ax.scatter(a1, i, s=29, color=C["whole"], edgecolor="white", lw=0.65, zorder=5)
        ax.scatter(a3, i, s=29, color=C["oracle"], edgecolor="white", lw=0.65, zorder=5)
        ax.text(0.965, i, D.signed(a3 - a1), fontsize=6.3, color=C["oracle"],
                fontweight="bold", ha="right", va="center")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([MDISP[m] for m in order], fontsize=6.0)
    ax.invert_yaxis()
    ax.set_xlim(0.30, 0.98)
    ax.set_xlabel("answer accuracy")
    ax.set_title("All six models degrade", fontsize=7.1, loc="left", pad=5)
    tidy(ax, "x")
    save(fig, "fig2-b")


# ---------------------------------------------------------------------------
# Figure 3 -- original panels, separated
# ---------------------------------------------------------------------------


def grounding_series():
    levels = sorted({D.demand(t) for t in D.GRND})
    answer, evidence, joint, counts = [], [], [], []
    for level in levels:
        sel = [g for g in D.GR if D.demand(g["task_id"]) == level]
        answer.append(np.mean([g["correct"] for g in sel]))
        evidence.append(np.mean([g["em"] for g in sel]))
        joint.append(np.mean([g["joint"] for g in sel]))
        counts.append(len({g["task_id"] for g in sel}))
    return levels, answer, evidence, joint, counts


def fig3_grounding() -> None:
    levels, answer, evidence, joint, counts = grounding_series()

    fig, ax = plt.subplots(figsize=(2.10, 2.12))
    fig.subplots_adjust(left=0.22, right=0.95, top=0.80, bottom=0.24)
    ax.fill_between(levels, evidence, answer, color=C["evid"], alpha=0.15, lw=0)
    ax.plot(levels, answer, "-o", color=C["whole"], lw=2.0, ms=4.0,
            markeredgecolor="white", markeredgewidth=0.55)
    ax.plot(levels, evidence, "-o", color=C["evid"], lw=2.0, ms=4.0,
            markeredgecolor="white", markeredgewidth=0.55)
    ax.plot(levels, joint, "--o", color=C["oracle"], lw=1.2, ms=2.8)
    ax.text(levels[0] + 0.08, answer[0] + 0.030, "answer", color=C["whole"],
            fontsize=6.0, fontweight="bold")
    ax.text(levels[0] + 0.08, evidence[0] + 0.030, "evidence", color=C["evid"],
            fontsize=6.0, fontweight="bold")
    ax.text(levels[0] + 0.08, joint[0] - 0.060, r"answer $\wedge$ evidence",
            color=C["oracle"], fontsize=5.5)
    ax.annotate("", xy=(levels[-1], evidence[-1]), xytext=(levels[-1], answer[-1]),
                arrowprops=dict(arrowstyle="<->", color=C["evid"], lw=1.0))
    ax.text(levels[-1] - 0.13, (answer[-1] + evidence[-1]) / 2,
            "grounding\ngap", color=C["evid"], fontsize=5.8,
            fontweight="bold", ha="right", va="center")
    ax.set_xticks(levels)
    ax.set_xticklabels([f"{d}\n$n$={n}" for d, n in zip(levels, counts)])
    ax.set_xlabel("visual evidence demand")
    ax.set_ylabel("accuracy")
    ax.set_ylim(-0.02, 0.78)
    ax.set_title("Answering holds; grounding does not", fontsize=7.0,
                 loc="left", pad=4)
    tidy(ax)
    save(fig, "fig3-a")

    fig, ax = plt.subplots(figsize=(2.75, 2.65))
    fig.subplots_adjust(left=0.24, right=0.97, top=0.70, bottom=0.23)
    for m, _ in MODELS:
        ys = [np.mean([g["em"] for g in D.GR
                       if g["model"] == m and D.demand(g["task_id"]) == level])
              for level in levels]
        ax.plot(levels, ys, "-o", color=MC[m], lw=1.15, ms=2.8,
                label=MDISP[m], markeredgewidth=0)
    ax.set_xticks(levels)
    ax.set_xlabel("visual evidence demand")
    ax.set_ylabel("evidence exact match")
    ax.set_ylim(-0.015, 0.72)
    ax.set_title("Every model shows the same collapse", fontsize=7.0,
                 loc="left", pad=32)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncols=3,
              handletextpad=0.18, columnspacing=0.55, labelspacing=0.12,
              borderaxespad=0.0, handlelength=0.65, fontsize=4.85)
    tidy(ax)
    save(fig, "fig3-b")

    fig, ax = plt.subplots(figsize=(1.52, 2.12))
    fig.subplots_adjust(left=0.30, right=0.94, top=0.80, bottom=0.24)
    conf = [g["conf"] for g in D.GR if isinstance(g["conf"], (int, float))]
    high = [g for g in D.GR if isinstance(g["conf"], (int, float)) and g["conf"] >= 0.9]
    vals = [np.mean([g["correct"] for g in high]), np.mean([g["em"] for g in high])]
    ax.bar([0, 1], [1, 1], color=T["paper"], width=0.62)
    ax.bar([0, 1], vals, color=[C["whole"], C["evid"]], width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v - 0.055 if v > 0.25 else v + 0.035, f"{v:.0%}",
                color="white" if v > 0.25 else C["evid"], fontsize=7.0,
                fontweight="bold", ha="center", va="top" if v > 0.25 else "bottom")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["answer\ncorrect", "evidence\nexact"], fontsize=5.6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("rate")
    ax.set_title("Confidence says nothing", fontsize=7.0, loc="left", pad=4)
    ax.text(0.5, 0.98, f"{len(high) / len(conf):.0%} report ≥ 0.9",
            transform=ax.transAxes, fontsize=5.5, color=C["mute"], ha="center", va="top")
    tidy(ax)
    save(fig, "fig3-c")


# ---------------------------------------------------------------------------
# Figure 4 -- original aesthetic, independent panels
# ---------------------------------------------------------------------------


def composition_groups():
    ids = sorted(MT.DATASET_IDS["layered_vqa_visual_canonical.json"])
    bins = [("≤2", lambda k: k <= 2), ("3", lambda k: k == 3),
            ("4", lambda k: k == 4), ("5", lambda k: k == 5)]
    return [(lab, [t for t in ids if fn(D.n_layers(t))]) for lab, fn in bins]


def fig4_dose() -> None:
    groups = composition_groups()
    x = np.arange(len(groups))
    whole = [D.acc_over("B1", g) for _, g in groups]
    parts = [D.acc_over("B6", g) for _, g in groups]

    fig, ax = plt.subplots(figsize=(3.05, 2.02))
    fig.subplots_adjust(left=0.18, right=0.96, top=0.81, bottom=0.24)
    ax.fill_between(x, parts, whole, color=C["parts"], alpha=0.15, lw=0)
    ax.plot(x, whole, "-o", color=C["whole"], lw=2.1, ms=4.2,
            markeredgecolor="white", markeredgewidth=0.6)
    ax.plot(x, parts, "-o", color=C["parts"], lw=2.1, ms=4.2,
            markeredgecolor="white", markeredgewidth=0.6)
    for i in x:
        ax.annotate("", xy=(i, parts[i]), xytext=(i, whole[i]),
                    arrowprops=dict(arrowstyle="<->", color=C["parts"], lw=0.8,
                                    alpha=0.70))
    ax.text(0.06, whole[0] + 0.030, "whole scene", color=C["whole"],
            fontsize=6.2, fontweight="bold")
    ax.text(0.06, parts[0] - 0.065, "same scene, in parts", color=C["parts"],
            fontsize=6.2, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n$n$={len(g)}" for lab, g in groups])
    ax.set_xlabel("layers the scene decomposes into")
    ax.set_ylabel("answer accuracy")
    ax.set_ylim(0.40, 0.76)
    ax.set_title("The whole holds steady; only the parts decline",
                 fontsize=7.1, loc="left", pad=4)
    tidy(ax)
    save(fig, "fig4-a")

    fig, ax = plt.subplots(figsize=(3.05, 2.02))
    fig.subplots_adjust(left=0.18, right=0.96, top=0.81, bottom=0.24)
    effects = [D.boot_gap("B6", "B1", g) for _, g in groups]
    ax.axhline(0, color=C["ink"], lw=0.8, zorder=3)
    ax.fill_between(x, [lo for _, lo, _ in effects],
                    [hi for _, _, hi in effects],
                    color=C["parts"], alpha=0.14, lw=0, zorder=1)
    ax.plot(x, [point for point, _, _ in effects], "-", color=C["parts"],
            lw=1.4, zorder=4)
    ax.errorbar(
        x, [point for point, _, _ in effects],
        yerr=[
            [point - lo for point, lo, _ in effects],
            [hi - point for point, _, hi in effects],
        ],
        fmt="o", color=C["parts"], ms=4.6, lw=1.3, capsize=2.4, zorder=6,
        markeredgecolor="white", markeredgewidth=0.6,
    )
    for i, (point, _, hi) in enumerate(effects):
        ax.text(i, hi + 0.012, D.signed(point), fontsize=6.0,
                color=C["parts"], ha="center", va="bottom", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for lab, _ in groups])
    ax.set_xlabel("layers the scene decomposes into")
    ax.set_ylabel("parts − whole")
    ax.set_ylim(-0.40, 0.10)
    ax.set_title("Composition cost grows with the pieces (95% CI)",
                 fontsize=7.1, loc="left", pad=4)
    tidy(ax)
    save(fig, "fig4-b")


def main() -> None:
    print(f"figures -> {OUT}")
    fig1_teaser()
    fig2_oracle()
    fig3_grounding()
    fig4_dose()

    # Figures 5 and 6 were already visually strong; regenerate those exact
    # plots into the new directory instead of restyling them unnecessarily.
    old_out = D.OUT
    try:
        D.OUT = OUT
        D.fig5_landscape()
        D.fig6_forest()
    finally:
        D.OUT = old_out


if __name__ == "__main__":
    main()
