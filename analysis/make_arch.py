#!/usr/bin/env python3
"""Draw the pipeline diagrams: simulation, post-hoc, and the two combined.

Run from anywhere:  python overleaf/make_arch.py
Writes PDF (vector, for LaTeX) and PNG (400 dpi) into `overleaf/figures/`.

Three variants are emitted so the paper can pick by the space it has:

    fig7_simulation.pdf   the conversation loop and how a conversation is graded
    fig8_posthoc.pdf      cross-judging and the seven behavioural analyses
    fig9_pipeline.pdf     corpus -> simulation -> analysis, all three bands

Design notes that matter for reading the output:

* **The loop is ours, not the one it is drawn after.** The reference diagram in
  the multi-turn text literature routes every turn through a strategy classifier
  and an answer extractor, and ends the conversation the moment an answer is
  judged correct. This pipeline does none of that: a conversation always runs to
  shard exhaustion and is graded exactly once, at the end, from the whole
  transcript. Drawing a per-turn verdict would misreport the experiment, so the
  diagram states the absence rather than leaving it to be inferred.
* **The subject is highlighted.** The evaluated assistant carries the only
  filled crimson box; every other node is apparatus.
* **The combined figure summarises, it does not repeat.** Grading collapses to
  one box in `fig9`; the two graders are separated only in `fig7`, where there
  is room to say what each one does.
* **Palette is kept in sync with `make_figs.py` by hand.** Importing it would
  drag in `make_tables`, which loads every log before it will draw a rectangle,
  and these diagrams need no data at all.
* **Geometry is isotropic.** Axes are unit-matched to the figure aspect and set
  to `equal`, so a rounded corner is a circle and a 12-wide box is the same
  measure whichever figure it sits in. Band-to-band connectors run in the white
  gutter between bands, which is why the bands are not flush.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

W = 5.5  # ICLR \textwidth in inches

C = {
    "whole": "#1B5C8F",
    "parts": "#C25E12",
    "recap": "#1F7A54",
    "oracle": "#A61E48",
    "evid": "#5A3E96",
    "ink": "#14171C",
    "mute": "#79808A",
}
T = {
    "whole": "#E8F0F7", "parts": "#FCEFE3", "recap": "#E6F3EC",
    "oracle": "#FAE9EE", "evid": "#EEEAF7", "mute": "#F2F0EC",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------

def canvas(h_in: float):
    fig, ax = plt.subplots(figsize=(W, h_in))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * h_in / W)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


class Box:
    __slots__ = ("x", "y", "w", "h")

    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2

    def r(self, f=0.5):
        return (self.x + self.w, self.y + self.h * f)

    def l(self, f=0.5):
        return (self.x, self.y + self.h * f)

    def t(self, f=0.5):
        return (self.x + self.w * f, self.y + self.h)

    def b(self, f=0.5):
        return (self.x + self.w * f, self.y)


def box(ax, x, y, w, h, text, ec, fc="white", fs=6.5, weight="bold",
        lw=1.0, r=1.7, z=6, tc=None):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc or ec, fontweight=weight, zorder=z + 1, linespacing=1.32)
    return Box(x, y, w, h)


def band(ax, x, y, w, h, color, r=2.2, z=0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=color, edgecolor="none", zorder=z))


def arrow(ax, p0, p1, color=C["ink"], lw=1.0, ls="-", z=8, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>,head_width=0.20,head_length=0.42",
        mutation_scale=9, color=color, lw=lw, linestyle=ls, zorder=z,
        shrinkA=1.5, shrinkB=1.5, connectionstyle=f"arc3,rad={rad}"))


def wire(ax, pts, color=C["ink"], lw=1.0, z=7):
    """A right-angled path with no head; heads are added separately."""
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=lw, zorder=z,
            solid_capstyle="round", solid_joinstyle="round")


def route(ax, pts, color=C["ink"], lw=1.0, z=8):
    wire(ax, pts[:-1], color, lw, z)
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, z=z)


def note(ax, x, y, text, color, bg, fs=6.0, weight="normal", ha="center",
         va="center", pad=0.38, z=20, style="normal"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, zorder=z,
            fontweight=weight, linespacing=1.32, style=style,
            bbox=dict(boxstyle=f"round,pad={pad}", facecolor=bg,
                      edgecolor=color, linewidth=0.7))


def label(ax, x, y, text, color=C["ink"], fs=5.9, ha="center", va="center",
          weight="normal", z=20, style="normal", rot=0):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, zorder=z,
            fontweight=weight, linespacing=1.28, style=style, rotation=rot)


def stage(ax, x, y, n, name, color, fs=7.2):
    ax.add_patch(FancyBboxPatch(
        (x, y - 2.0), 4.0, 4.0, boxstyle="round,pad=0,rounding_size=2.0",
        facecolor=color, edgecolor="none", zorder=21))
    ax.text(x + 2.0, y, str(n), ha="center", va="center", fontsize=6.4,
            color="white", fontweight="bold", zorder=22)
    ax.text(x + 5.4, y, name, ha="left", va="center", fontsize=fs,
            color=color, fontweight="bold", zorder=22)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", facecolor="white")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# --------------------------------------------------------------------------
# shared sub-drawings
# --------------------------------------------------------------------------

def judge_matrix(ax, x, y, cell=2.05, n=6, gap=0.40, caption=True):
    """A 6x6 assistant-by-judge grid with the primary-run diagonal picked out."""
    for i in range(n):
        for j in range(n):
            on = i == j
            ax.add_patch(Rectangle(
                (x + j * (cell + gap), y + (n - 1 - i) * (cell + gap)), cell, cell,
                facecolor=C["evid"] if on else T["evid"],
                edgecolor=C["evid"] if on else "#CFC8E4",
                linewidth=0.55, zorder=7))
    side = n * cell + (n - 1) * gap
    b = Box(x, y, side, side)
    if caption:
        label(ax, b.cx, y - 2.6, "assistant × judge", C["evid"], 5.6, weight="bold")
        label(ax, b.cx, y - 5.4, "diagonal = the primary runs", C["evid"], 5.4,
              style="italic")
    return b


def loop_row(ax, y, h, fs=6.4, sample_text="Sample\nquestion · shards\nlayers · schedule",
             caption_below=False):
    """The reveal/respond loop. Returns the boxes and the loop-back height."""
    smp = box(ax, 1.0, y, 19.5, h, sample_text, C["whole"], T["whole"], fs=fs)
    usr = box(ax, 27.5, y, 19.0, h, "User Agent\nscripted | LLM",
              C["parts"], T["parts"], fs=fs)
    ast = box(ax, 53.5, y, 19.5, h, "Evaluated\nAssistant",
              C["oracle"], "#F6D9E1", fs=fs + 0.5, lw=1.5)
    trn = box(ax, 79.5, y, 19.5, h, "Turn appended\nto transcript",
              C["mute"], "white", fs=fs, tc=C["ink"])

    arrow(ax, smp.r(), usr.l(), C["mute"])
    arrow(ax, usr.r(), ast.l(), C["parts"], lw=1.2)
    arrow(ax, ast.r(), trn.l(), C["oracle"], lw=1.2)
    label(ax, (usr.x + usr.w + ast.x) / 2, y + h + 2.4,
          "reveal $\\leq$1 shard\n+ its layers", C["parts"], 5.7, weight="bold")
    label(ax, (ast.x + ast.w + trn.x) / 2, y + h + 2.4, "response",
          C["oracle"], 5.7, weight="bold")

    top = y + h + 7.2
    route(ax, [trn.t(), (trn.cx, top), (usr.cx, top), usr.t()], C["parts"])
    label(ax, (trn.cx + usr.cx) / 2 + (7.0 if caption_below else 0.0),
          top + (-2.2 if caption_below else 2.0),
          "unrevealed shards remain  ·  no early stop", C["parts"], 5.8,
          weight="bold")
    return smp, usr, ast, trn, top


# --------------------------------------------------------------------------
# Figure 7 -- simulation only
# --------------------------------------------------------------------------

def fig_simulation():
    H = 3.15
    fig, ax = canvas(H)
    band(ax, 0.0, 28.5, 100, 28.8, T["parts"])
    band(ax, 0.0, 1.0, 100, 24.0, T["evid"])

    stage(ax, 1.6, 53.4, 1, "Sharded conversation", C["parts"])
    smp, usr, ast, trn, top = loop_row(ax, 32.5, 11.0)

    stage(ax, 1.6, 21.2, 2, "Graded once, at the end", C["evid"])

    lay = box(ax, 30.0, 12.6, 27.0, 8.0,
              "Layer-set scoring\ngold · minimal · distractors",
              C["evid"], "white", fs=6.2)
    jdg = box(ax, 30.0, 2.6, 27.0, 8.0,
              "Answer Judge\nLLM equivalence", C["evid"], T["evid"], fs=6.2)
    rec = box(ax, 64.0, 7.3, 18.0, 8.6, "Conversation\nrecord",
              C["mute"], "white", fs=6.2, tc=C["ink"])

    # transcript -> both graders, in parallel
    wire(ax, [trn.b(), (trn.cx, 25.8), (25.0, 25.8), (25.0, jdg.cy)], C["evid"])
    arrow(ax, (25.0, lay.cy), lay.l(), C["evid"])
    arrow(ax, (25.0, jdg.cy), jdg.l(), C["evid"])
    label(ax, 62.0, 27.3, "shards exhausted  →  the whole transcript",
          C["evid"], 5.8, weight="bold")
    label(ax, lay.cx, lay.y + lay.h + 1.6, "only when the reply lists layers",
          C["mute"], 5.4, style="italic")

    # both graders -> one record
    arrow(ax, lay.r(), rec.l(0.80), C["evid"])
    arrow(ax, jdg.r(), rec.l(0.20), C["evid"])

    note(ax, 12.5, 11.0,
         "no turn classifier,\nno answer extractor,\nno early stop",
         C["evid"], "white", fs=6.0, weight="bold")
    note(ax, 85.0, 20.4,
         "RECAP replays a finished sharded\nlog, grading only the recap turn\n"
         "SNOWBALL re-attaches every\nrevealed layer each turn",
         C["mute"], "white", fs=5.4, pad=0.34)
    save(fig, "fig7_simulation")


# --------------------------------------------------------------------------
# Figure 8 -- post-hoc only
# --------------------------------------------------------------------------

ANALYSES = [
    ("cross-judge", "self-preference\nresidual"),
    ("first-attempt", "when it commits,\nwhat that costs"),
    ("verbosity", "length vs.\ncorrectness"),
    ("bloat", "length of the\n$n$-th attempt"),
    ("citation", "which turn's layer\nit cites"),
    ("grounding", "answered before\nevidence arrived"),
    ("churn", "changed · abandoned\n· recovered"),
]


def fig_posthoc():
    H = 3.30
    fig, ax = canvas(H)
    band(ax, 0.0, 32.0, 100, 27.0, T["recap"])
    band(ax, 0.0, 1.0, 100, 28.0, T["mute"])

    stage(ax, 1.6, 55.5, 1, "Post-hoc re-judging", C["recap"])

    logs = box(ax, 1.5, 39.0, 21.0, 12.5,
               "Conversation logs\n3 runs × 6 models\n× 16 conditions",
               C["mute"], "white", fs=6.2, tc=C["ink"])
    mat = judge_matrix(ax, 31.0, 38.8)
    rej = box(ax, 55.5, 39.0, 20.0, 12.5,
              "Re-graded\nevery conversation,\nevery judge",
              C["recap"], T["recap"], fs=6.2)
    per = box(ax, 79.0, 39.0, 20.0, 12.5,
              "Per-conversation\nis_correct · attempts\nfirst-correct turn",
              C["recap"], "white", fs=6.0)

    arrow(ax, logs.r(), (mat.x - 1.2, logs.cy), C["mute"])
    arrow(ax, (mat.x + mat.w + 1.2, logs.cy), rej.l(), C["evid"])
    arrow(ax, rej.r(), per.l(), C["recap"])

    stage(ax, 1.6, 25.8, 2, "Seven analyses", C["ink"])

    n = len(ANALYSES)
    gap, x0 = 1.4, 1.5
    w = (99.0 - x0 - gap * (n - 1)) / n
    spine = 21.5
    for i, (name, sub) in enumerate(ANALYSES):
        x = x0 + i * (w + gap)
        first = i == 0
        b = box(ax, x, 7.0, w, 12.0, "", C["evid"] if first else C["ink"],
                "white", lw=1.5 if first else 0.8)
        ax.text(b.cx, b.y + b.h - 3.0, name, ha="center", va="center", fontsize=6.2,
                color=C["evid"] if first else C["ink"], fontweight="bold", zorder=9)
        ax.text(b.cx, b.y + 4.6, sub, ha="center", va="center", fontsize=5.7,
                color=C["ink"], zorder=9, linespacing=1.30)
        arrow(ax, (b.cx, spine), b.t(), C["mute"], lw=0.8)
    wire(ax, [(x0 + w / 2, spine), (99.0 - w / 2, spine)], C["mute"], lw=0.9)
    wire(ax, [per.b(), (per.cx, spine)], C["mute"], lw=0.9)

    note(ax, 50.0, 3.3,
         "the cross-judge residual corrects every headline number;  "
         "the other six characterise $\\it{how}$ the two losses happen",
         C["ink"], "white", fs=5.8, pad=0.36)
    save(fig, "fig8_posthoc")


# --------------------------------------------------------------------------
# Figure 9 -- the whole pipeline
# --------------------------------------------------------------------------

def fig_pipeline():
    H = 4.95
    fig, ax = canvas(H)
    B1, B2, B3 = (65.0, 24.0), (28.0, 33.0), (1.0, 24.0)
    band(ax, 0.0, B1[0], 100, B1[1], T["whole"])
    band(ax, 0.0, B2[0], 100, B2[1], T["parts"])
    band(ax, 0.0, B3[0], 100, B3[1], T["recap"])

    # ---- band 1: corpus ---------------------------------------------------
    stage(ax, 1.6, 85.0, 1, "Corpus", C["whole"])
    y1, h1 = 68.5, 11.5
    sc = box(ax, 1.5, y1, 20.0, h1, "Layered scenes\ncomposite +\nRGBA layer stack",
             C["whole"], "white", fs=6.0)
    qs = box(ax, 25.5, y1, 19.0, h1, "300 questions\nauthored\n6 types",
             C["whole"], "white", fs=6.0)
    an = box(ax, 48.5, y1, 24.0, h1,
             "Shards + evidence\nLLM draft, then\nhand-corrected in full",
             C["whole"], T["whole"], fs=6.0)
    ex = box(ax, 76.5, y1, 22.5, h1, "8 dataset exports\n→ 16 conditions",
             C["whole"], "white", fs=6.0)
    for a, b in [(sc, qs), (qs, an), (an, ex)]:
        arrow(ax, a.r(), b.l(), C["whole"])
    label(ax, 50.0, 66.0,
          "gold layers · minimal sufficient sets · shard→layer support · distractors",
          C["whole"], 5.5, style="italic")

    # band 1 -> band 2, routed through the gutter
    route(ax, [ex.b(), (ex.cx, 63.0), (11.25, 63.0), (11.25, 61.6)], C["whole"])
    label(ax, 44.0, 64.4, "one conversation per question × condition",
          C["whole"], 5.5, style="italic")

    # ---- band 2: simulation ----------------------------------------------
    stage(ax, 1.6, 58.4, 2, "Simulation", C["parts"])
    smp, usr, ast, trn, top = loop_row(
        ax, 43.0, 10.5, fs=6.2, sample_text="Sample\nshards · layers\nschedule",
        caption_below=True)

    grd = box(ax, 24.0, 30.5, 44.0, 8.5,
              "Graded once, at the end, from the whole transcript\n"
              "answer judge  ·  layer-set scoring",
              C["evid"], T["evid"], fs=6.1)
    rec = box(ax, 74.0, 30.5, 25.0, 8.5, "Conversation record",
              C["mute"], "white", fs=6.1, tc=C["ink"])
    route(ax, [trn.b(), (trn.cx, 41.0), (grd.cx, 41.0), grd.t()], C["evid"])
    arrow(ax, grd.r(), rec.l(), C["evid"])
    label(ax, 12.0, 34.7, "no turn classifier,\nno early stop", C["evid"], 5.6,
          weight="bold")

    # band 2 -> band 3, routed through the gutter

    # ---- band 3: analysis -------------------------------------------------
    stage(ax, 1.6, 22.8, 3, "Analysis", C["recap"], fs=7.0)
    y3, h3 = 8.0, 11.0
    mat = judge_matrix(ax, 5.0, y3 + 0.5, cell=1.55, gap=0.34, caption=False)
    label(ax, mat.cx, y3 - 1.4, "assistant × judge", C["evid"], 5.4, weight="bold")
    label(ax, mat.cx, y3 - 3.8, "diagonal = primary runs", C["evid"], 5.2,
          style="italic")
    rej = box(ax, 24.0, y3, 23.0, h3,
              "Post-hoc re-judging\nevery judge × every model",
              C["recap"], T["recap"], fs=6.0)
    res = box(ax, 50.5, y3, 22.0, h3, "Self-preference\nresidual",
              C["recap"], "white", fs=6.0)
    beh = box(ax, 76.0, y3, 23.0, h3, "Six behavioural\nanalyses", C["recap"],
              "white", fs=6.0)
    route(ax, [rec.b(), (rec.cx, 26.2), (rej.cx, 26.2), rej.t()], C["mute"])
    arrow(ax, (mat.x + mat.w + 1.2, rej.cy), rej.l(), C["evid"])
    arrow(ax, rej.r(), res.l(), C["recap"])
    arrow(ax, res.r(), beh.l(), C["recap"])
    label(ax, res.cx, y3 - 2.4, "corrects every headline number", C["recap"], 5.4,
          style="italic")
    label(ax, beh.cx, y3 - 2.4,
          "first-attempt · verbosity · bloat\ncitation · grounding · churn",
          C["recap"], 5.2, style="italic")
    save(fig, "fig9_pipeline")


def main():
    print(f"diagrams -> {OUT}")
    fig_simulation()
    fig_posthoc()
    fig_pipeline()


if __name__ == "__main__":
    main()
