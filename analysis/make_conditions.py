#!/usr/bin/env python3
"""Draw the delivery-condition chart: all fourteen conditions in one figure.

Run from anywhere:  python overleaf/make_conditions.py
Writes `figures/fig10_conditions.{pdf,png}`.

Design notes that matter for reading the output:

* **Two lanes, not one.** The text-only multi-turn literature has a single
  channel to draw, so its condition chart is one column of shards per turn.
  Every condition here delivers on two channels at once, and which channel is
  fragmented is the paper's central contrast, so each panel is split into a
  question lane and an image lane.
* **The grouping is the argument.** The three blocks are "question fragmented",
  "image fragmented", and "both", which is exactly the comparison the results
  turn on. Reading the figure top to bottom is reading the paper's claim.
* **Shade carries support, not decoration.** A question shard and the layers
  that support it share a shade. Aligned and misaligned then differ visibly
  while delivering identical counts, which no amount of arrow-drawing would
  show as directly.
* **A tinted lane means "still available".** The full image persists through a
  sharded-question conversation and the full question persists through a
  sharded-image one; a tinted lane says so without spending a glyph per turn.
"""

from __future__ import annotations

from matplotlib.patches import FancyBboxPatch, Rectangle

from make_arch import C, T, canvas, label, save

# --------------------------------------------------------------------------
# glyph vocabulary
# --------------------------------------------------------------------------

Q_RAMP = ["#C3C8EA", "#959EDB", "#6570C6", "#3B44A6"]   # question shards
L_RAMP = ["#B4DED6", "#79C4B7", "#40A493", "#178071"]   # supporting layers
LAYER = "#4CAE9C"          # a layer when support is not the manipulation
DIST_F, DIST_E = "#E7E5E1", "#A8A39A"   # distractor layer
FULL_Q, FULL_I = "#2C3590", "#0F6A5D"   # the whole question, the whole image
BAND_Q, BAND_I = "#EDEEFA", "#E4F2EF"   # "still available" lane tints
INK_Q, INK_I = "#1B2270", "#0A4A40"     # numerals inside the glyphs

TURNS = 5


def q_bar(ax, x, y, w, h, shade=None, z=6):
    """A question part. Full width and solid when it is the whole question."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.45",
        facecolor=FULL_Q if shade is None else Q_RAMP[shade % 4],
        edgecolor=FULL_Q, linewidth=0.5, zorder=z))


def layer_sq(ax, x, y, s, shade=None, kind="layer", z=6):
    """One layer image. Distractors are hollow, evidence is filled."""
    if kind == "dist":
        fc, ec, ls = DIST_F, DIST_E, (0, (1.4, 1.0))
    elif shade is None:
        fc, ec, ls = LAYER, "#0F6A5D", "-"
    else:
        fc, ec, ls = L_RAMP[shade % 4], "#0F6A5D", "-"
    ax.add_patch(FancyBboxPatch(
        (x, y), s, s, boxstyle="round,pad=0,rounding_size=0.28",
        facecolor=fc, edgecolor=ec, linewidth=0.55, linestyle=ls, zorder=z))


def full_img(ax, x, y, w, h, z=6):
    """The undivided composite: wider, solid, with a seam so it reads as one."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.45",
        facecolor=FULL_I, edgecolor="#0A4A40", linewidth=0.5, zorder=z))
    ax.plot([x + w * 0.62, x + w * 0.62], [y + 0.18, y + h - 0.18],
            color="#CDE9E3", lw=0.5, zorder=z + 1)


# --------------------------------------------------------------------------
# conditions
# --------------------------------------------------------------------------
# Each turn is "question|image".
#   question:  Q whole question,  0-3 shard with that shade
#   image:     I whole image,  l a layer,  e0-e3 a layer supporting that shard,
#              d a distractor layer
CONDITIONS = [
    # the question is fragmented, the image stays whole
    ("FULL", ["Q|I"], None),
    ("CONCAT-Q", ["0123|I"], None),
    ("SHARDED-Q", ["0|I", "1|", "2|", "3|"], "img"),
    ("RECAP-Q", ["0|I", "1|", "2|", "3|", "Q|I"], "img"),
    ("SNOWBALL-Q", ["0|I", "01|", "012|", "0123|"], "img"),
    ("SHUFFLED-CONCAT", ["0213|I"], None),
    # the image is fragmented, the question stays whole
    ("CONCAT-V", ["Q|llll"], None),
    ("SHARDED-V", ["Q|l", "|l", "|l", "|l"], "txt"),
    ("RECAP-V", ["Q|l", "|l", "|l", "|l", "|I"], "txt"),
    ("SNOWBALL-V", ["Q|l", "|ll", "|lll", "|llll"], "txt"),
    ("EVIDENCE-FIRST", ["Q|l", "|l", "|d", "|d"], "txt"),
    ("EVIDENCE-LAST", ["Q|d", "|d", "|l", "|l"], "txt"),
    # both are fragmented
    ("ALIGNED-VQ", ["0|e0", "1|e1", "2|e2", "3|e3"], None),
    ("MISALIGNED-VQ", ["0|e2", "1|e3", "2|e0", "3|e1"], None),
]

BLOCKS = [
    (0, 6, "The question arrives in parts, the image stays whole",
     "#2C3590", "#F2F3FB"),
    (6, 12, "The image arrives in parts, the question stays whole",
     "#0F6A5D", "#EDF6F3"),
    (12, 14, "Both arrive in parts, bound or not", "#37424F", "#F2F3F4"),
]

# geometry, in the isotropic 0-100 canvas
PW, GAP, X0 = 14.6, 1.72, 2.0
ROW, HDR = 3.05, 6.2
LANE_Q, LANE_I, PAD = 5.5, 7.6, 0.6
BLOCK_H = HDR + TURNS * ROW


def parse(turn: str):
    q, _, im = turn.partition("|")
    items = []
    i = 0
    while i < len(im):
        ch = im[i]
        if ch == "e" and i + 1 < len(im):
            items.append(("e", int(im[i + 1])))
            i += 2
        else:
            items.append((ch, None))
            i += 1
    return q, items


def draw_panel(ax, x, y, name, turns, persist, accent, captions=False,
               numbered=False):
    """One condition. `y` is the bottom of the panel."""
    qx = x + PAD
    ix = x + PAD + LANE_Q + PAD

    label(ax, x + PW / 2, y + BLOCK_H - 2.0, name, accent, 5.5, weight="bold")
    if captions:
        label(ax, qx + LANE_Q / 2, y + BLOCK_H - 4.7, "question", C["mute"], 4.6)
        label(ax, ix + LANE_I / 2, y + BLOCK_H - 4.7, "image", C["mute"], 4.6)

    top = y + TURNS * ROW
    used = len(turns) * ROW - 0.4
    if persist == "img":
        ax.add_patch(Rectangle((ix - 0.3, top - used - 0.4), LANE_I + 0.6, used,
                               facecolor=BAND_I, edgecolor="none", zorder=1))
    elif persist == "txt":
        ax.add_patch(Rectangle((qx - 0.3, top - used - 0.4), LANE_Q + 0.6, used,
                               facecolor=BAND_Q, edgecolor="none", zorder=1))

    for t in range(TURNS):
        yy = top - (t + 1) * ROW
        ax.plot([x + 0.2, x + PW - 0.2], [yy - 0.05, yy - 0.05],
                color="#E6E2DA", lw=0.4, zorder=2)
        if t >= len(turns):
            continue
        q, items = parse(turns[t])
        h = ROW * 0.56
        cy = yy + (ROW - h) / 2

        if q == "Q":
            q_bar(ax, qx, cy, LANE_Q, h)
        elif q:
            n = len(q)
            bw = (LANE_Q - 0.35 * (n - 1)) / n
            for k, ch in enumerate(q):
                q_bar(ax, qx + k * (bw + 0.35), cy, bw, h, shade=int(ch))
                if numbered:
                    ax.text(qx + k * (bw + 0.35) + bw / 2, cy + h / 2,
                            str(int(ch) + 1), ha="center", va="center",
                            fontsize=4.2, color=INK_Q, zorder=9)

        s = min(h, 1.75)
        sy = yy + (ROW - s) / 2
        if items and items[0][0] == "I":
            full_img(ax, ix, cy, LANE_I, h)
        else:
            for k, (kind, shade) in enumerate(items):
                sx = ix + k * (s + 0.42)
                layer_sq(ax, sx, sy, s,
                         shade=shade,
                         kind="dist" if kind == "d" else "layer")
                if numbered and shade is not None:
                    ax.text(sx + s / 2, sy + s / 2, str(shade + 1), ha="center",
                            va="center", fontsize=4.2, color=INK_I, zorder=9)


def legend(ax, x, y, w, h):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
        facecolor="white", edgecolor="#CFC9BE", linewidth=0.7, zorder=5))
    cx, cy = x + 2.6, y + h - 3.4
    rows = [
        ("qfull", "the whole question"),
        ("qpart", "one question part, shaded by its order"),
        ("ifull", "the whole image"),
        ("layer", "one layer of the image"),
        ("shade", "a layer numbered by the question part it supports"),
        ("dist", "a layer that supports no part of the question"),
        ("band", "still available from an earlier turn"),
    ]
    for i, (kind, text) in enumerate(rows):
        yy = cy - i * 3.0
        if kind == "qfull":
            q_bar(ax, cx, yy, 6.0, 1.5)
        elif kind == "qpart":
            for k in range(3):
                q_bar(ax, cx + k * 2.15, yy, 1.8, 1.5, shade=k)
        elif kind == "ifull":
            full_img(ax, cx, yy, 6.0, 1.5)
        elif kind == "layer":
            for k in range(3):
                layer_sq(ax, cx + k * 2.15, yy - 0.1, 1.7)
        elif kind == "shade":
            for k in range(3):
                layer_sq(ax, cx + k * 2.15, yy - 0.1, 1.7, shade=k)
                ax.text(cx + k * 2.15 + 0.85, yy + 0.75, str(k + 1), ha="center",
                        va="center", fontsize=4.2, color=INK_I, zorder=9)
        elif kind == "dist":
            for k in range(3):
                layer_sq(ax, cx + k * 2.15, yy - 0.1, 1.7, kind="dist")
        else:
            ax.add_patch(Rectangle((cx, yy - 0.3), 6.0, 2.0,
                                   facecolor=BAND_I, edgecolor="none", zorder=6))
        label(ax, cx + 8.0, yy + 0.75, text, C["ink"], 5.3, ha="left")


def fig_conditions():
    n_blocks = len(BLOCKS)
    total = n_blocks * BLOCK_H + (n_blocks - 1) * 4.2 + 3.0
    H = 5.5 * total / 100.0
    fig, ax = canvas(H)

    for bi, (lo, hi, title, accent, tint) in enumerate(BLOCKS):
        y = total - 3.0 - (bi + 1) * BLOCK_H - bi * 4.2
        n = hi - lo
        ax.add_patch(FancyBboxPatch(
            (X0 - 1.2, y - 1.4), n * (PW + GAP) - GAP + 2.4, BLOCK_H + 1.6,
            boxstyle="round,pad=0,rounding_size=1.6",
            facecolor=tint, edgecolor="none", zorder=0))
        label(ax, X0 - 0.6, y + BLOCK_H + 1.6, title, accent, 6.6, ha="left",
              weight="bold")

        for i, (name, turns, persist) in enumerate(CONDITIONS[lo:hi]):
            draw_panel(ax, X0 + i * (PW + GAP), y, name, turns, persist, accent,
                       captions=(i == 0), numbered=(bi == 2))

        # turn numbers, once per block, outside the first panel
        top = y + TURNS * ROW
        deepest = max(len(t) for _, t, _ in CONDITIONS[lo:hi])
        for t in range(deepest):
            label(ax, X0 - 2.0, top - (t + 0.5) * ROW, str(t + 1), C["mute"], 4.8)
        label(ax, X0 - 2.0, top + 1.4, "turn", C["mute"], 4.8, style="italic")

        if bi == n_blocks - 1:
            lx = X0 + n * (PW + GAP) + 1.5
            legend(ax, lx, y - 1.4, 100 - lx - 1.5, BLOCK_H + 1.6)

    save(fig, "fig10_conditions")


if __name__ == "__main__":
    print(f"conditions chart -> figures/")
    fig_conditions()
