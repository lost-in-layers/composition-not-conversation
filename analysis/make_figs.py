#!/usr/bin/env python3
"""Regenerate every figure in the paper from the Final-* simulation logs.

Run from anywhere:  python overleaf/make_figs.py
Writes PDF (vector, for LaTeX) and PNG (300 dpi, for looking at) into
`overleaf/figures/`.

Design notes that matter for reading the output:

* **Numbers come from `make_tables`.** This module imports it rather than
  re-implementing the loaders, so a figure and the table beside it cannot drift:
  same three runs, same six models, same convention that a conversation missing
  from the logs (context-window overflow) counts as wrong rather than vanishing.
* **Photographs are the dataset's own.** Every image drawn here is a file the
  models were actually sent -- the composite from `selected_1200/`, the flattened
  layer renders from `lic_renders/`. Nothing is illustrated or redrawn, so the
  reader is looking at the same pixels the model was. Every tick and cross is
  the majority verdict over 18 model-runs, printed by `--audit`.
* **One palette, one meaning.** Blue is an intact scene, amber is the same
  information in pieces, green is a scene restored, crimson is certified
  evidence with the scene removed, purple is grounding. A colour never changes
  what it denotes between figures, and each figure sits on the tint of whatever
  it is about.
* **The claim is annotated, not left to be inferred.** Every panel names its
  effect on the panel, in the colour of the thing that caused it.
* **Uncertainty is drawn, never implied.** Spread across the three runs is an
  error bar; contrasts carry question-clustered bootstrap intervals from
  `make_tables.pooled_contrast`, which is also what the tables report.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch
from PIL import Image

import make_tables as MT

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
DATA = MT.DATA
RUNS, MODELS, MODEL_IDS = MT.RUNS, MT.MODELS, MT.MODEL_IDS

# --------------------------------------------------------------------------
# house style
# --------------------------------------------------------------------------

W = 5.5  # ICLR \textwidth in inches; figures are drawn at final size

C = {
    "whole": "#1B5C8F",   # the scene, intact
    "parts": "#C25E12",   # the same information, in pieces
    "recap": "#1F7A54",   # the scene restored
    "oracle": "#A61E48",  # certified evidence, scene removed
    "evid": "#5A3E96",    # grounding / evidence identification
    "ink": "#14171C",
    "mute": "#79808A",
    "grid": "#E3DFD7",
    "ok": "#1F7A54",
    "bad": "#B02037",
}
# Washes. Light enough to print, strong enough to zone a figure at a glance.
T = {
    "whole": "#E8F0F7", "parts": "#FCEFE3", "recap": "#E6F3EC",
    "oracle": "#FAE9EE", "evid": "#EEEAF7", "paper": "#F7F5F1",
}

# Okabe-Ito, colour-blind safe. Order matches MT.MODELS (descending accuracy).
MODEL_C = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00", "#56B4E9"]
MC = {m: MODEL_C[i] for i, (m, _) in enumerate(MODELS)}
MDISP = dict(MODELS)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.2,
    "axes.titlesize": 7.8,
    "axes.labelsize": 7.2,
    "xtick.labelsize": 6.7,
    "ytick.labelsize": 6.7,
    "legend.fontsize": 6.4,
    "legend.frameon": False,
    "axes.edgecolor": "#B9B3A8",
    "axes.linewidth": 0.6,
    "axes.labelcolor": C["ink"],
    "axes.facecolor": "none",
    "figure.facecolor": "white",
    "text.color": C["ink"],
    "xtick.color": C["ink"],
    "ytick.color": C["ink"],
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": C["grid"],
    "grid.linewidth": 0.5,
    "lines.solid_capstyle": "round",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", facecolor="white")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def tidy(ax, grid="y") -> None:
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, linewidth=0.5, color=C["grid"])
    ax.tick_params(length=2.2, pad=1.6)


def zone(fig, x0, y0, x1, y1, color, r=0.014, ec="none", lw=0.0, z=-10):
    """A rounded tinted region in figure coordinates, drawn behind everything."""
    fig.patches.append(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={r}",
        transform=fig.transFigure, facecolor=color, edgecolor=ec,
        linewidth=lw, zorder=z, clip_on=False))


def pill(ax, x, y, text, fg, bg, fs=7.0, weight="bold", tf=None, ha="center",
         va="center", pad=0.34, alpha=1.0):
    """A callout the reader cannot miss: the claim, on the colour that caused it."""
    return ax.text(x, y, text, transform=tf if tf is not None else ax.transAxes,
                   ha=ha, va=va, fontsize=fs, color=fg, fontweight=weight,
                   zorder=30, linespacing=1.35,
                   bbox=dict(boxstyle=f"round,pad={pad}", facecolor=bg,
                             edgecolor=fg, linewidth=0.7, alpha=alpha))


def badge(ax, x, y, ok=True, r=0.055, lw=1.25):
    """A drawn tick/cross. Drawn, not typed, because Times has neither glyph."""
    col = C["ok"] if ok else C["bad"]
    ax.add_patch(Circle((x, y), r, transform=ax.transAxes, facecolor=col,
                        edgecolor="white", linewidth=0.9, zorder=20, clip_on=False))
    k = r * 0.52
    if ok:
        pts = [(x - k, y + k * 0.05), (x - k * 0.25, y - k * 0.62), (x + k, y + k * 0.72)]
        ax.add_line(Line2D([p[0] for p in pts], [p[1] for p in pts],
                           transform=ax.transAxes, color="white", lw=lw,
                           solid_capstyle="round", zorder=21, clip_on=False))
    else:
        for dx in (1, -1):
            ax.add_line(Line2D([x - k * 0.70 * dx, x + k * 0.70 * dx],
                               [y - k * 0.70, y + k * 0.70],
                               transform=ax.transAxes, color="white", lw=lw,
                               solid_capstyle="round", zorder=21, clip_on=False))


def drop_arrow(ax, x0, y0, x1, y1, color, lw=1.1, rad=0.0, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>,head_width=0.17,head_length=0.34",
                                color=color, lw=lw, linestyle=ls,
                                shrinkA=2.5, shrinkB=2.5,
                                connectionstyle=f"arc3,rad={rad}"), zorder=12)


def signed(x, nd=3):
    return ("+" if x >= 0 else "−") + f"{abs(x):.{nd}f}".lstrip("0")


def pts(x):
    return ("+" if x >= 0 else "−") + f"{abs(x) * 100:.0f}"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def _load(fn):
    return {r["task_id"]: r for r in json.loads((DATA / fn).read_text(encoding="utf-8"))}


CANON = _load("layered_vqa_visual_canonical.json")
GRND = _load("layered_vqa_grounding.json")

CATS = ["direct_ask", "attribute_binding", "counting", "spatial",
        "occlusion", "multi_hop_reference"]
CAT_DISP = {
    "direct_ask": "direct ask", "attribute_binding": "attribute binding",
    "counting": "counting", "spatial": "spatial",
    "occlusion": "occlusion", "multi_hop_reference": "multi-hop reference",
}
CAT_SHORT = {
    "direct_ask": "direct ask", "attribute_binding": "attribute",
    "counting": "counting", "spatial": "spatial",
    "occlusion": "occlusion", "multi_hop_reference": "multi-hop",
}
# Which categories need the scene rather than an object. Set once, used to
# split figure 2 and to justify the wording in its annotation.
RELATIONAL = {"counting", "spatial", "occlusion", "multi_hop_reference"}


def img_path(rel: str) -> Path:
    return (DATA / rel).resolve()


def n_layers(tid: str) -> int:
    return len(CANON[tid]["meta"]["layer_labels"])


def cat(tid: str) -> str:
    return MT.QMETA[tid]["category"]


def demand(tid: str) -> int:
    return MT.QMETA[tid]["demand"]


def load_grounding() -> list[dict]:
    """Per-conversation Experiment-A2 records, keyed enough to slice by question."""
    rows = []
    for run in RUNS:
        d = MT.LOGS / run / "layered_vqa" / "full"
        if not d.is_dir():
            continue
        for path in d.glob("*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("dataset_fn") != "layered_vqa_grounding.json":
                    continue
                if rec.get("assistant_model") not in MODEL_IDS:
                    continue
                for e in rec["trace"]:
                    c = e.get("content")
                    if e.get("role") != "log" or not isinstance(c, dict):
                        continue
                    if c.get("type") != "answer-evaluation":
                        continue
                    ev = c.get("evaluation_return", {})
                    lm = ev.get("layer_metrics") or {}
                    ls = ev.get("layer_selection") or {}
                    rows.append({
                        "run": run, "model": rec["assistant_model"],
                        "task_id": rec["task_id"],
                        "correct": bool(ev.get("is_correct")),
                        "em": bool(lm.get("exact_match")),
                        "joint": bool(ev.get("joint_correct")),
                        "conf": ls.get("confidence"),
                        "pred": ls.get("layer_ids"),
                    })
    return rows


GR = load_grounding()


def macro(key: str) -> float:
    return float(np.mean([MT.acc(m, key) for m in MODEL_IDS]))


def acc_over(key: str, tids) -> float:
    tids = [t for t in tids if t in MT.DATASET_IDS[MT.KEY_FN[key]]]
    if not tids:
        return float("nan")
    hits = tot = 0
    for m in MODEL_IDS:
        for r in RUNS:
            v = MT.scored(r, m, key)
            hits += sum(v[t] for t in tids)
            tot += len(tids)
    return hits / tot


def boot_gap(key_a: str, key_b: str, tids, n_boot=2000, seed=0):
    """Cluster bootstrap of acc(a) - acc(b) over a question subset."""
    tids = [t for t in tids
            if t in MT.DATASET_IDS[MT.KEY_FN[key_a]]
            and t in MT.DATASET_IDS[MT.KEY_FN[key_b]]]
    per_q = {t: [MT.scored(r, m, key_a)[t] - MT.scored(r, m, key_b)[t]
                 for m in MODEL_IDS for r in RUNS] for t in tids}
    if not per_q:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean([v for d in per_q.values() for v in d]))
    rnd = random.Random(seed)
    keys = list(per_q)
    out = []
    for _ in range(n_boot):
        s = n = 0
        for _ in range(len(keys)):
            d = per_q[keys[rnd.randrange(len(keys))]]
            s += sum(d)
            n += len(d)
        out.append(s / n)
    out.sort()
    return point, out[int(0.025 * n_boot)], out[int(0.975 * n_boot)]


def hits(tid: str, key: str) -> int:
    return sum(MT.scored(r, m, key)[tid] for m, _ in MODELS for r in RUNS)


# --------------------------------------------------------------------------
# image helpers
# --------------------------------------------------------------------------

def _fit(im: Image.Image, box, bg=(255, 255, 255)) -> Image.Image:
    im = im.convert("RGB")
    im.thumbnail(box, Image.LANCZOS)
    canvas = Image.new("RGB", box, bg)
    canvas.paste(im, ((box[0] - im.width) // 2, (box[1] - im.height) // 2))
    return canvas


def _frame(sheet, x, y, w, h, rgb, k=2):
    W_, H_ = sheet.size
    for t in range(k):
        for xx in range(x - t, x + w + t):
            for yy in (y - t, y + h + t - 1):
                if 0 <= xx < W_ and 0 <= yy < H_:
                    sheet.putpixel((xx, yy), rgb)
        for yy in range(y - t, y + h + t):
            for xx in (x - t, x + w + t - 1):
                if 0 <= xx < W_ and 0 <= yy < H_:
                    sheet.putpixel((xx, yy), rgb)


def montage(paths, cols, cell=(190, 140), gap=7, bg=(255, 255, 255),
            edge=(196, 191, 182), highlight=None, hl_rgb=(166, 30, 72)):
    """Tile layer renders into one image, so a strip of layers costs one axes."""
    paths = [p for p in paths if Path(p).exists()]
    rows = max(1, int(np.ceil(len(paths) / cols)))
    Wpx = cols * cell[0] + (cols + 1) * gap
    Hpx = rows * cell[1] + (rows + 1) * gap
    sheet = Image.new("RGB", (Wpx, Hpx), bg)
    for i, p in enumerate(paths):
        r, c = divmod(i, cols)
        sheet.paste(_fit(Image.open(p), cell),
                    (gap + c * (cell[0] + gap), gap + r * (cell[1] + gap)))
        rgb = hl_rgb if (highlight and i in highlight) else edge
        _frame(sheet, gap + c * (cell[0] + gap), gap + r * (cell[1] + gap),
               cell[0], cell[1], rgb, k=3 if (highlight and i in highlight) else 1)
    return sheet


def show(ax, im, frame=None, lw=1.1):
    ax.imshow(np.asarray(im))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(frame is not None)
        if frame:
            s.set_color(frame)
            s.set_linewidth(lw)
    for s in ("top", "right"):
        ax.spines[s].set_visible(frame is not None)
    return ax


def layer_files(tid: str, ids=None):
    d = DATA.parent / "lic_renders" / CANON[tid]["meta"]["scene_id"]
    if ids is None:
        ids = sorted(int(k) for k in CANON[tid]["meta"]["layer_labels"])
    return [d / f"layer_{i}.png" for i in ids]


def composite(tid: str) -> Path:
    return img_path(GRND[tid]["images"][0]["path"])


# --------------------------------------------------------------------------
# Figure 1 -- teaser
# --------------------------------------------------------------------------

# Hero examples, chosen because the logs say so, not because they read well.
HERO = "lvis_000000299103::spatial"
HERO_G = "lvis_000000108442::direct_ask"
HERO_G_GOLD = [2]   # bulb
HERO_G_PRED = [4]   # zebra -- 6 of 18 model-runs answered "Blue" citing this


def fig1_teaser() -> None:
    fig = plt.figure(figsize=(W, 5.55))
    gs = gridspec.GridSpec(
        4, 3, figure=fig,
        height_ratios=[1.00, 0.74, 0.34, 1.66],
        hspace=0.36, wspace=0.46,
        left=0.085, right=0.978, top=0.845, bottom=0.062)

    fig.text(0.5, 0.975, "Vision–language models lose the scene, not the thread",
             ha="center", va="center", fontsize=10.4, color=C["ink"],
             fontweight="bold")
    fig.text(0.5, 0.944,
             "the same information, re-shaped three ways — and what each re-shaping costs",
             ha="center", va="center", fontsize=6.6, color=C["mute"], style="italic")

    heads = [
        ("Loss in composition", C["parts"], T["parts"],
         "the whole scene, delivered in pieces"),
        ("Oracle inversion", C["oracle"], T["oracle"],
         "only the evidence that entails the answer"),
        ("Loss in grounding", C["evid"], T["evid"],
         "the right answer, the wrong evidence"),
    ]

    meta = CANON[HERO]["meta"]
    q, a = CANON[HERO]["question"], CANON[HERO]["answer"]
    gold = meta["gold_layer_ids"]
    all_ids = sorted(int(k) for k in meta["layer_labels"])
    axes_by_col = [[], [], []]

    def add(r, c):
        ax = fig.add_subplot(gs[r, c])
        axes_by_col[c].append(ax)
        return ax

    # --- row 0: the intact scene, three times -------------------------------
    for j, tid in enumerate([HERO, HERO, HERO_G]):
        ax = add(0, j)
        show(ax, _fit(Image.open(composite(tid)), (430, 300)), frame=C["whole"])
        ax.set_title("whole scene" if j < 2 else "whole scene, answer correct",
                     fontsize=6.5, color=C["whole"], pad=2.4)
        badge(ax, 0.93, 0.10, ok=True)

    # --- row 1: the same information, re-shaped ------------------------------
    ax = add(1, 0)
    show(ax, montage(layer_files(HERO, [0] + all_ids), cols=3, cell=(150, 112)),
         frame=C["parts"])
    ax.set_title("its layers — same information", fontsize=6.5,
                 color=C["parts"], pad=2.4)
    badge(ax, 0.945, 0.14, ok=False)

    ax = add(1, 1)
    show(ax, montage(layer_files(HERO, gold), cols=2, cell=(190, 140),
                     highlight={0, 1}), frame=C["oracle"])
    lbl = " + ".join(meta["layer_labels"][str(i)] for i in gold)
    ax.set_title(f"gold evidence only: {lbl}", fontsize=6.5, color=C["oracle"], pad=2.4)
    badge(ax, 0.945, 0.15, ok=False)

    gmeta = CANON[HERO_G]["meta"]
    glab = gmeta["layer_labels"]
    ax = add(1, 2)
    show(ax, montage(layer_files(HERO_G, HERO_G_GOLD + HERO_G_PRED), cols=2,
                     cell=(190, 140), highlight={1}), frame=C["evid"])
    ax.set_title("gold evidence   |   what it cited", fontsize=6.5,
                 color=C["evid"], pad=2.4)
    badge(ax, 0.945, 0.15, ok=False)

    # --- row 2: one caption per column, boxed so nothing bleeds sideways -----
    caps = [
        (f"“{q}”\n→ {a}", C["parts"], T["parts"]),
        ("given the cat and the mouse and nothing else,\n"
         "it can no longer say which is to the right", C["oracle"], T["oracle"]),
        (f"“{CANON[HERO_G]['question']}” → {CANON[HERO_G]['answer']}\n"
         f"correct — while citing the {glab[str(HERO_G_PRED[0])]}",
         C["evid"], T["evid"]),
    ]
    for j, (txt, col, bg) in enumerate(caps):
        ax = add(2, j)
        ax.axis("off")
        pill(ax, 0.5, 0.5, txt, col, bg, fs=6.0, weight="normal", pad=0.42)

    # --- plot A: the composition ladder --------------------------------------
    ax = add(3, 0)
    keys = ["B1", "B6", "B7", "B8"]
    xt = ["whole", "parts,\n1 turn", "parts,\nper turn", "then\nwhole"]
    for m, _ in MODELS:
        ax.plot(range(4), [MT.acc(m, k) for k in keys], "-", color=MC[m], lw=0.9,
                alpha=0.55, zorder=2)
        ax.scatter(range(4), [MT.acc(m, k) for k in keys], s=8, color=MC[m],
                   alpha=0.8, zorder=3, linewidths=0)
    ys = [macro(k) for k in keys]
    ax.plot(range(4), ys, "-o", color=C["ink"], lw=2.2, ms=4.6, zorder=6,
            markeredgecolor="white", markeredgewidth=0.7)
    ax.set_xticks(range(4))
    ax.set_xticklabels(xt, fontsize=5.8)
    ax.set_xlim(-0.45, 3.45)
    ax.set_ylim(0.30, 0.955)
    ax.set_ylabel("answer accuracy", fontsize=6.6)
    drop_arrow(ax, 0.05, ys[0] - 0.012, 0.95, ys[1] + 0.012, C["parts"], lw=1.3)
    pill(ax, 0.30, 0.905, f"{pts(ys[1] - ys[0])} pts", C["parts"], T["parts"], fs=7.4,
         tf=ax.transAxes)
    drop_arrow(ax, 2.05, ys[2] + 0.012, 2.95, ys[3] - 0.012, C["recap"], lw=1.3)
    pill(ax, 0.80, 0.905, "restored", C["recap"], T["recap"], fs=7.0, tf=ax.transAxes)
    ax.text(0.5, 0.055, "6 models × 3 runs; bold line = mean",
            transform=ax.transAxes, fontsize=5.5, color=C["mute"], ha="center")
    tidy(ax)

    # --- plot B: oracle inversion by category --------------------------------
    ax = add(3, 1)
    ids = sorted(MT.DATASET_IDS["layered_vqa_oracle_gold.json"])
    rows = []
    for c in CATS:
        t = [i for i in ids if cat(i) == c]
        rows.append((c, acc_over("A1", t), acc_over("A3", t)))
    rows.sort(key=lambda r: r[2] - r[1])
    ax.axhspan(-0.7, 3.5, color=T["oracle"], zorder=0)
    for i, (c, a1, a3) in enumerate(rows):
        drop_arrow(ax, a1, i, a3, i, C["oracle"] if c in RELATIONAL else C["mute"],
                   lw=1.15)
        ax.scatter([a1], [i], s=24, color=C["whole"], zorder=13,
                   edgecolor="white", linewidth=0.6)
        ax.scatter([a3], [i], s=24, color=C["oracle"], zorder=13,
                   edgecolor="white", linewidth=0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([CAT_SHORT[c] for c, *_ in rows], fontsize=6.1)
    ax.set_xlabel("answer accuracy", fontsize=6.6)
    ax.set_xlim(0.30, 0.82)
    ax.set_ylim(-0.7, len(rows) - 0.05)
    ax.text(0.795, 1.6, "needs\nthe scene", fontsize=5.9, color=C["oracle"],
            ha="right", va="center", fontweight="bold", linespacing=1.25)
    ax.text(0.795, 4.62, "needs\nthe object", fontsize=5.9, color=C["mute"],
            ha="right", va="center", linespacing=1.25)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=C["whole"], ms=3.6, label="whole image"),
        Line2D([], [], marker="o", ls="", color=C["oracle"], ms=3.6, label="evidence only"),
    ], loc="lower left", bbox_to_anchor=(-0.30, 1.0), ncols=2, handletextpad=0.25,
        columnspacing=0.8, borderaxespad=0.0, handlelength=0.8, fontsize=5.9)
    tidy(ax, grid="x")

    # --- plot C: grounding falls faster than answering ------------------------
    ax = add(3, 2)
    lv = sorted({demand(t) for t in GRND})
    A = [np.mean([g["correct"] for g in GR if demand(g["task_id"]) == d]) for d in lv]
    E = [np.mean([g["em"] for g in GR if demand(g["task_id"]) == d]) for d in lv]
    ax.fill_between(lv, E, A, color=C["evid"], alpha=0.16, lw=0, zorder=1)
    ax.plot(lv, A, "-o", color=C["whole"], lw=2.0, ms=4.0, zorder=4,
            markeredgecolor="white", markeredgewidth=0.6)
    ax.plot(lv, E, "-o", color=C["evid"], lw=2.0, ms=4.0, zorder=4,
            markeredgecolor="white", markeredgewidth=0.6)
    ax.text(lv[0] + 0.06, A[0] + 0.045, "answer", fontsize=6.2, color=C["whole"],
            fontweight="bold")
    ax.text(lv[0] + 0.06, E[0] + 0.045, "evidence", fontsize=6.2, color=C["evid"],
            fontweight="bold")
    ax.set_xticks(lv)
    ax.set_xlabel("visual evidence demand", fontsize=6.6)
    ax.set_ylabel("accuracy", fontsize=6.6)
    ax.set_ylim(-0.03, 0.96)
    pill(ax, 0.5, 0.905, f"answer {pts(A[-1] - A[0])} pts,\n"
                         f"evidence {E[0] / max(E[-1], 1e-9):.0f}× collapse",
         C["evid"], T["evid"], fs=6.6, tf=ax.transAxes)
    ax.annotate("", xy=(lv[-1] - 0.10, E[-1] + 0.02), xytext=(lv[-1] - 0.10, A[-1] - 0.02),
                arrowprops=dict(arrowstyle="<->", color=C["evid"], lw=0.9), zorder=12)
    ax.text(lv[-1] - 0.17, (A[-1] + E[-1]) / 2, "grounding\ngap", fontsize=5.9,
            color=C["evid"], ha="right", va="center", linespacing=1.2)
    tidy(ax)

    # --- tinted zones, sized from where the axes actually landed -------------
    fig.canvas.draw()
    for j, (name, col, tint, sub) in enumerate(heads):
        bb = [a.get_position() for a in axes_by_col[j]]
        x0 = min(b.x0 for b in bb) - (0.070 if j == 0 else 0.058)
        x1 = max(b.x1 for b in bb) + 0.016
        zone(fig, x0, 0.018, x1, 0.918, tint)
        xc = (x0 + x1) / 2
        fig.text(xc, 0.902, name, ha="center", va="center", fontsize=8.8,
                 color=col, fontweight="bold")
        fig.text(xc, 0.874, sub, ha="center", va="center", fontsize=6.1,
                 color=C["mute"], style="italic")
    save(fig, "fig1_teaser")


# --------------------------------------------------------------------------
# Figure 2 -- oracle inversion
# --------------------------------------------------------------------------

def fig2_oracle() -> None:
    fig = plt.figure(figsize=(W, 2.62))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.30, 1.0],
                           wspace=0.46, left=0.165, right=0.978,
                           top=0.845, bottom=0.155)
    ids = sorted(MT.DATASET_IDS["layered_vqa_oracle_gold.json"])

    ax = fig.add_subplot(gs[0, 0])
    rows = []
    for c in CATS:
        t = [i for i in ids if cat(i) == c]
        p, lo, hi = boot_gap("A3", "A1", t)
        rows.append((c, acc_over("A1", t), acc_over("A3", t), len(t), p))
    rows.sort(key=lambda r: r[4])
    n_rel = sum(1 for r in rows if r[0] in RELATIONAL)
    ax.axhspan(-0.7, n_rel - 0.5, color=T["oracle"], zorder=0)
    for i, (c, a1, a3, n, p) in enumerate(rows):
        col = C["oracle"] if c in RELATIONAL else C["mute"]
        drop_arrow(ax, a1, i, a3, i, col, lw=1.35)
        ax.scatter([a1], [i], s=30, color=C["whole"], zorder=13,
                   edgecolor="white", linewidth=0.7)
        ax.scatter([a3], [i], s=30, color=C["oracle"], zorder=13,
                   edgecolor="white", linewidth=0.7)
        ax.text(0.815, i, signed(p), fontsize=6.6, color=col, ha="right",
                va="center", fontweight="bold" if c in RELATIONAL else "normal")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{CAT_DISP[c]}  ({n})" for c, _, _, n, _ in rows], fontsize=6.5)
    ax.set_xlabel("answer accuracy")
    ax.set_xlim(0.30, 0.82)
    ax.set_ylim(-0.7, len(rows) - 0.2)
    ax.text(0.815, n_rel - 0.75, "$\\Delta$", fontsize=6.6, color=C["ink"],
            ha="right", va="bottom", fontweight="bold")
    ax.text(0.312, n_rel - 1.5, "questions about arrangements", fontsize=6.2,
            color=C["oracle"], ha="left", va="center", fontweight="bold")
    ax.text(0.312, n_rel + 0.5, "questions about objects", fontsize=6.2,
            color=C["mute"], ha="left", va="center")
    ax.set_title("(a) removing the scene hurts — and hurts relations most",
                 fontsize=7.3, loc="left", pad=15)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=C["whole"], ms=4, label="whole image"),
        Line2D([], [], marker="o", ls="", color=C["oracle"], ms=4,
               label="gold evidence only"),
    ], loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=2, handletextpad=0.25,
        columnspacing=1.0, borderaxespad=0.0, handlelength=0.8, fontsize=6.1)
    tidy(ax, grid="x")

    ax = fig.add_subplot(gs[0, 1])
    order = sorted(MODEL_IDS, key=lambda m: MT.acc(m, "A3") - MT.acc(m, "A1"))
    for i, m in enumerate(order):
        a1, a3 = MT.acc(m, "A1"), MT.acc(m, "A3")
        drop_arrow(ax, a1, i, a3, i, C["oracle"], lw=1.35)
        ax.scatter([a1], [i], s=30, color=C["whole"], zorder=13,
                   edgecolor="white", linewidth=0.7)
        ax.scatter([a3], [i], s=30, color=C["oracle"], zorder=13,
                   edgecolor="white", linewidth=0.7)
        ax.text(0.905, i, signed(a3 - a1), fontsize=6.6, color=C["oracle"],
                ha="right", va="center", fontweight="bold")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([MDISP[m] for m in order], fontsize=6.5)
    ax.set_xlabel("answer accuracy")
    ax.set_xlim(0.30, 0.92)
    ax.set_ylim(-0.7, len(order) - 0.2)
    ax.set_title("(b) six of six, no exceptions", fontsize=7.3, loc="left", pad=4)
    tidy(ax, grid="x")
    save(fig, "fig2_oracle_inversion")


# --------------------------------------------------------------------------
# Figure 3 -- loss in grounding
# --------------------------------------------------------------------------

def fig3_grounding() -> None:
    fig = plt.figure(figsize=(W, 2.30))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.05, 0.94, 0.92],
                           wspace=0.42, left=0.082, right=0.985,
                           top=0.845, bottom=0.225)
    lv = sorted({demand(t) for t in GRND})

    ax = fig.add_subplot(gs[0, 0])
    A, E, J, N = [], [], [], []
    for d in lv:
        sel = [g for g in GR if demand(g["task_id"]) == d]
        A.append(np.mean([g["correct"] for g in sel]))
        E.append(np.mean([g["em"] for g in sel]))
        J.append(np.mean([g["joint"] for g in sel]))
        N.append(len({g["task_id"] for g in sel}))
    ax.fill_between(lv, E, A, color=C["evid"], alpha=0.15, lw=0, zorder=1)
    ax.plot(lv, A, "-o", color=C["whole"], lw=2.1, ms=4.2, zorder=5,
            markeredgecolor="white", markeredgewidth=0.6, label="answer accuracy")
    ax.plot(lv, E, "-o", color=C["evid"], lw=2.1, ms=4.2, zorder=5,
            markeredgecolor="white", markeredgewidth=0.6, label="evidence exact match")
    ax.plot(lv, J, "--o", color=C["oracle"], lw=1.2, ms=2.8, zorder=4, label="joint")
    ax.text(lv[0] + 0.08, J[0] - 0.075, "answer $\\wedge$ evidence", fontsize=5.9,
            color=C["oracle"], ha="left", va="center")
    ax.annotate("", xy=(lv[-1] - 0.03, E[-1]), xytext=(lv[-1] - 0.03, A[-1]),
                arrowprops=dict(arrowstyle="<->", color=C["evid"], lw=1.0), zorder=12)
    ax.text(lv[-1] - 0.14, (A[-1] + E[-1]) / 2, "grounding\ngap", fontsize=6.0,
            color=C["evid"], ha="right", va="center", linespacing=1.2,
            fontweight="bold")
    ax.set_xticks(lv)
    ax.set_xticklabels([f"{d}\n$n$={n}" for d, n in zip(lv, N)])
    ax.set_xlabel("visual evidence demand", labelpad=1)
    ax.set_ylabel("accuracy")
    ax.set_xlim(lv[0] - 0.20, lv[-1] + 1.45)
    ax.set_ylim(-0.03, 0.80)
    pill(ax, lv[-1] + 0.22, A[-1], f"answer\n{pts(A[-1] - A[0])} pts", C["whole"],
         T["whole"], fs=6.2, tf=ax.transData, ha="left")
    pill(ax, lv[-1] + 0.22, E[-1] + 0.10, f"evidence\n{E[0] / max(E[-1], 1e-9):.0f}× collapse",
         C["evid"], T["evid"], fs=6.2, tf=ax.transData, ha="left")
    ax.set_title("(a) answering holds; grounding does not", fontsize=7.1,
                 loc="left", pad=4)
    tidy(ax)

    ax = fig.add_subplot(gs[0, 1])
    for m, _ in MODELS:
        ys = [np.mean([g["em"] for g in GR
                       if g["model"] == m and demand(g["task_id"]) == d]) for d in lv]
        ax.plot(lv, ys, "-o", color=MC[m], lw=1.3, ms=3.0, label=MDISP[m],
                markeredgewidth=0, zorder=4)
    ax.set_xticks(lv)
    ax.set_xlabel("visual evidence demand", labelpad=1)
    ax.set_ylabel("evidence exact match")
    ax.set_ylim(-0.015, 0.72)
    ax.set_title("(b) every model, same collapse", fontsize=7.1, loc="left", pad=4)
    ax.legend(loc="upper right", ncols=2, handletextpad=0.2, columnspacing=0.5,
              labelspacing=0.15, borderaxespad=0.0, handlelength=0.75, fontsize=5.5)
    tidy(ax)

    ax = fig.add_subplot(gs[0, 2])
    conf = [g["conf"] for g in GR if isinstance(g["conf"], (int, float))]
    hi = [g for g in GR if isinstance(g["conf"], (int, float)) and g["conf"] >= 0.9]
    n, bins, patches = ax.hist(conf, bins=np.arange(0, 1.0001, 0.1),
                               color=C["mute"], edgecolor="white", linewidth=0.6)
    patches[-1].set_facecolor(C["evid"])
    ax.set_xlabel("self-reported confidence", labelpad=1)
    ax.set_ylabel("replies")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, max(n) * 1.42)
    pill(ax, 0.5, 0.965,
         f"{len(hi) / max(len(conf), 1):.0%} of replies claim $\\geq$0.9\n"
         f"they are right {np.mean([g['correct'] for g in hi]):.0%} of the time,\n"
         f"and grounded {np.mean([g['em'] for g in hi]):.0%} of the time",
         C["evid"], T["evid"], fs=6.0, weight="normal", tf=ax.transAxes, va="top")
    ax.set_title("(c) confidence says nothing", fontsize=7.1, loc="left", pad=4)
    tidy(ax)
    save(fig, "fig3_loss_in_grounding")


# --------------------------------------------------------------------------
# Figure 4 -- composition scales with the number of pieces
# --------------------------------------------------------------------------

def fig4_dose() -> None:
    fig = plt.figure(figsize=(W, 2.20))
    gs = gridspec.GridSpec(1, 2, wspace=0.33, left=0.088, right=0.982,
                           top=0.845, bottom=0.215, figure=fig)
    ids = sorted(MT.DATASET_IDS["layered_vqa_visual_canonical.json"])
    bins = [("≤2", lambda k: k <= 2), ("3", lambda k: k == 3),
            ("4", lambda k: k == 4), ("5", lambda k: k == 5)]
    groups = [(lab, [t for t in ids if f(n_layers(t))]) for lab, f in bins]
    x = np.arange(len(groups))

    ax = fig.add_subplot(gs[0, 0])
    whole = [acc_over("B1", g) for _, g in groups]
    parts = [acc_over("B6", g) for _, g in groups]
    ax.fill_between(x, parts, whole, color=C["parts"], alpha=0.16, lw=0, zorder=1)
    ax.plot(x, whole, "-o", color=C["whole"], lw=2.2, ms=4.4, zorder=5,
            markeredgecolor="white", markeredgewidth=0.6)
    ax.plot(x, parts, "-o", color=C["parts"], lw=2.2, ms=4.4, zorder=5,
            markeredgecolor="white", markeredgewidth=0.6)
    ax.text(0.06, whole[0] + 0.030, "whole scene", fontsize=6.4, color=C["whole"],
            fontweight="bold")
    ax.text(0.06, parts[0] - 0.095, "same scene, in parts", fontsize=6.4,
            color=C["parts"], fontweight="bold")
    for i in x:
        ax.annotate("", xy=(i, parts[i]), xytext=(i, whole[i]),
                    arrowprops=dict(arrowstyle="<->", color=C["parts"], lw=0.8,
                                    alpha=0.65), zorder=6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n$n$={len(g)}" for lab, g in groups])
    ax.set_xlabel("layers the scene decomposes into", labelpad=1)
    ax.set_ylabel("answer accuracy")
    ax.set_ylim(0.34, 0.85)
    pill(ax, 0.5, 0.955, "the whole scene holds steady;\nonly the parts decline",
         C["parts"], T["parts"], fs=6.4, tf=ax.transAxes, va="top")
    ax.set_title("(a) busier scenes are not simply harder", fontsize=7.1,
                 loc="left", pad=4)
    tidy(ax)

    ax = fig.add_subplot(gs[0, 1])
    pt = [boot_gap("B6", "B1", g) for _, g in groups]
    ax.axhline(0, color=C["ink"], lw=0.8, zorder=3)
    ax.fill_between(x, [lo for _, lo, _ in pt], [hi for _, _, hi in pt],
                    color=C["parts"], alpha=0.14, lw=0, zorder=1)
    ax.plot(x, [p for p, _, _ in pt], "-", color=C["parts"], lw=1.4, zorder=4)
    ax.errorbar(x, [p for p, _, _ in pt],
                yerr=[[p - lo for p, lo, _ in pt], [hi - p for p, _, hi in pt]],
                fmt="o", color=C["parts"], ms=4.6, lw=1.3, capsize=2.4, zorder=6,
                markeredgecolor="white", markeredgewidth=0.6)
    for i in x:
        ax.text(i, pt[i][2] + 0.012, signed(pt[i][0]), fontsize=6.0,
                color=C["parts"], ha="center", va="bottom", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for lab, _ in groups])
    ax.set_xlabel("layers the scene decomposes into", labelpad=1)
    ax.set_ylabel("parts $-$ whole")
    ax.set_ylim(-0.40, 0.10)
    ax.set_title("(b) the composition cost grows with the pieces (95% CI)",
                 fontsize=7.1, loc="left", pad=4)
    tidy(ax)
    save(fig, "fig4_composition_dose")


# --------------------------------------------------------------------------
# Figure 5 -- the delivery landscape
# --------------------------------------------------------------------------

FAM_C = {"Text": C["whole"], "Visual": C["parts"],
         "Layered": C["evid"], "Order": C["recap"]}
FAM_T = {"Text": T["whole"], "Visual": T["parts"],
         "Layered": T["evid"], "Order": T["recap"]}
# Family glosses live in the caption, not the gutter: rotated, they are taller
# than a two-row group and collide with the condition labels.
FAM_NOTE = {"Text": "the question in pieces", "Visual": "the image in pieces",
            "Layered": "both, bound or not", "Order": "same pieces, new order"}
COND_DISP = {
    "B1": "FULL", "B2": "CONCAT-Q", "B3": "SHARDED-Q", "B4": "RECAP-Q",
    "B5": "SNOWBALL-Q", "B6": "CONCAT-V", "B7": "SHARDED-V", "B8": "RECAP-V",
    "B9": "SNOWBALL-V", "B10": "ALIGNED-VQ", "B11": "MISALIGNED-VQ",
    "B12": "EVIDENCE-FIRST", "B13": "EVIDENCE-LAST", "B14": "SHUFFLED-CONCAT",
}


def fig5_landscape() -> None:
    fig, ax = plt.subplots(figsize=(W, 3.35))
    fig.subplots_adjust(left=0.262, right=0.905, top=0.905, bottom=0.115)
    rows = [(k, fam) for k, _, _, _, fam in MT.EXP_B]

    y, pos, prev = [], 0.0, None
    for k, fam in rows:
        if prev is not None and fam != prev:
            pos += 0.62
        y.append(pos)
        prev = fam
        pos += 1.0

    ref = macro("B1")
    fams = ["Text", "Visual", "Layered", "Order"]
    for fam in fams:
        yy = [y[i] for i, (_, f) in enumerate(rows) if f == fam]
        ax.axhspan(min(yy) - 0.5, max(yy) + 0.5, color=FAM_T[fam], zorder=0)
        ax.text(-0.268, (min(yy) + max(yy)) / 2, fam,
                transform=ax.get_yaxis_transform(), fontsize=7.6,
                color=FAM_C[fam], rotation=90, ha="center", va="center",
                fontweight="bold")

    ax.axvline(ref, color=C["ink"], lw=0.9, ls=(0, (4, 2)), zorder=6)
    ax.text(ref, -1.15, "FULL", fontsize=6.3, color=C["ink"], ha="center",
            va="bottom", fontweight="bold")

    for (k, fam), yy in zip(rows, y):
        v = macro(k)
        ax.barh(yy, v, height=0.66, color=FAM_C[fam], alpha=0.30, zorder=2,
                edgecolor="none")
        ax.plot([v, v], [yy - 0.33, yy + 0.33], color=FAM_C[fam], lw=2.2, zorder=7)
        for m, _ in MODELS:
            ax.scatter(MT.acc(m, k), yy, s=9, color=MC[m], zorder=8, alpha=0.95,
                       edgecolor="white", linewidth=0.35)
        d = v - ref
        ax.text(1.005, yy, signed(d) if k != "B1" else "—",
                transform=ax.get_yaxis_transform(), fontsize=6.3, va="center",
                ha="left", fontweight="bold" if abs(d) > 0.05 else "normal",
                color=C["oracle"] if d < -0.05 else (C["mute"] if d < 0.02 else C["recap"]))
    ax.text(1.005, -1.15, "", transform=ax.get_yaxis_transform(),
            fontsize=6.0, va="bottom", ha="left", color=C["ink"], fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([COND_DISP[k] for k, _ in rows], fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlim(0.30, 0.92)
    ax.set_ylim(max(y) + 0.75, -1.30)
    ax.set_xlabel("answer accuracy   (bar = mean over 6 models, dots = each model)",
                  labelpad=2)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=MC[m], ms=3.8,
                              label=MDISP[m]) for m, _ in MODELS],
              loc="lower left", bbox_to_anchor=(0.0, 1.005), ncols=6,
              handletextpad=0.25, columnspacing=0.75, borderaxespad=0.0,
              handlelength=0.9)
    tidy(ax, grid="x")
    save(fig, "fig5_delivery_landscape")


# --------------------------------------------------------------------------
# Figure 6 -- every claim, with an interval
# --------------------------------------------------------------------------

def fig6_forest() -> None:
    fig, ax = plt.subplots(figsize=(W, 2.55))
    fig.subplots_adjust(left=0.315, right=0.840, top=0.885, bottom=0.175)

    rows = []
    for name, _tex, cond, ref, _gloss in MT.CONTRASTS:
        p, lo, hi, pv, agree = MT.pooled_contrast(cond, ref)
        rows.append((name, p, lo, hi, pv, agree))
    rows.sort(key=lambda r: r[1])

    ax.axvspan(-0.30, 0, color=T["oracle"], zorder=0, alpha=0.55)
    ax.axvspan(0, 0.30, color=T["recap"], zorder=0, alpha=0.55)
    ax.axvline(0, color=C["ink"], lw=0.9, zorder=6)

    for i, (name, p, lo, hi, pv, agree) in enumerate(rows):
        sig = not (lo <= 0 <= hi)
        col = (C["oracle"] if p < 0 else C["recap"]) if sig else C["mute"]
        ax.plot([lo, hi], [i, i], color=col, lw=2.0, solid_capstyle="round", zorder=4)
        ax.scatter([p], [i], s=40, color=col, zorder=7, edgecolor="white",
                   linewidth=0.8)
        ax.text(1.02, i, signed(p), transform=ax.get_yaxis_transform(),
                fontsize=6.5, color=col, ha="right", va="center",
                fontweight="bold" if sig else "normal")
        ax.text(1.16, i, f"{agree}/6", transform=ax.get_yaxis_transform(),
                fontsize=6.0, color=C["mute"], ha="right", va="center")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlabel("change in answer accuracy   "
                  "(question-clustered bootstrap, 95% CI)", labelpad=2)
    ax.set_xlim(-0.235, 0.155)
    ax.set_ylim(len(rows) - 0.4, -1.25)
    ax.text(-0.005, -1.15, "delivery hurts", fontsize=6.6, color=C["oracle"],
            ha="right", va="bottom", fontweight="bold")
    ax.text(0.005, -1.15, "delivery helps", fontsize=6.6, color=C["recap"],
            ha="left", va="bottom", fontweight="bold")
    ax.text(1.02, -1.15, "$\\Delta$", transform=ax.get_yaxis_transform(),
            fontsize=6.5, color=C["ink"], ha="right", va="bottom", fontweight="bold")
    ax.text(1.16, -1.15, "models", transform=ax.get_yaxis_transform(),
            fontsize=6.0, color=C["mute"], ha="right", va="bottom")
    tidy(ax, grid="x")
    save(fig, "fig6_contrasts")


# --------------------------------------------------------------------------

def audit() -> None:
    """Print the verdicts every tick and cross in figure 1 stands on."""
    print("  figure-1 hero verdicts (correct model-runs out of 18):")
    for lbl, key in [("whole scene", "B1"), ("layers, one turn", "B6"),
                     ("layers, per turn", "B7"), ("layers then whole", "B8"),
                     ("gold evidence only", "A3")]:
        print(f"    {HERO:44} {lbl:20} {hits(HERO, key):2}/18")
    n = sum(1 for g in GR if g["task_id"] == HERO_G and g["correct"]
            and g["pred"] == HERO_G_PRED)
    print(f"    {HERO_G:44} {'correct, cited zebra':20} {n:2}/18")


def main() -> None:
    print(f"figures -> {OUT}")
    print(f"  grounding records: {len(GR)}  "
          f"({len({g['task_id'] for g in GR})} questions x "
          f"{len(MODEL_IDS)} models x {len(RUNS)} runs)")
    audit()
    fig2_oracle()
    fig3_grounding()
    fig4_dose()
    fig5_landscape()
    fig6_forest()
    fig1_teaser()


if __name__ == "__main__":
    if "--audit" in sys.argv:
        audit()
    else:
        main()
