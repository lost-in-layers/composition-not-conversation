"""Worked examples: a scene beside its layers, and the question beside its shards.

Two outputs. fig_examples.tex is the main-paper figure and shows three scenes
chosen for depth and for category spread. tab_layer_scale.tex is the appendix
table and walks the whole range of scene depth, from one object over a
background up to five.

Tiles are cut here rather than in LaTeX so the gold and distractor layers can
carry their own border colour. A gold layer gets the paper teal, a distractor
gets grey, and the caption repeats the same distinction in words.

Usage: python overleaf/make_gallery.py
"""
import json
import re
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
LIC = ROOT / "data" / "derived" / "lic" / "layered_vqa_aligned_vq.json"
RENDERS = ROOT / "data" / "derived" / "lic_renders"
SCENES = ROOT / "data" / "selected_1200"
OUT = Path(__file__).parent
GAL = OUT / "gallery"
B = chr(92)

GOLD = "#2F7C8C"
GREY = "#C2C2C2"
EDGE = "#ABABAB"
SCENE_W = 760
TILE_W = 460

# Three scenes for the main figure. Deep scenes, one per mechanism: an
# occlusion that only exists in the composite, a count whose members are one
# per layer, and a chain of references that needs four layers at once.
MAIN = ["lvis_000000299103::occlusion",
        "lvis_000000360982::counting",
        "lvis_000000232801::multi_hop_reference"]

# One scene per depth, for the appendix.
SCALE = ["laion_000111264::direct_ask",
         "lvis_000000367092::spatial",
         "lvis_000000500507::counting",
         "lvis_000000418717::multi_hop_reference",
         "laion_000372123::counting"]

CATNAME = {"direct_ask": "Direct ask", "attribute_binding": "Attribute binding",
           "occlusion": "Occlusion", "counting": "Counting",
           "spatial": "Spatial", "multi_hop_reference": "Multi-hop reference"}


def tex(s):
    """Make a string safe to drop into a LaTeX cell."""
    s = s.replace("�", "'").replace("’", "'")
    s = s.replace("“", "``").replace("”", "''")
    for a, b in [("\\", B + "textbackslash "), ("&", B + "&"), ("%", B + "%"),
                 ("$", B + "$"), ("#", B + "#"), ("_", B + "_"),
                 ("{", B + "{"), ("}", B + "}"), ("~", B + "textasciitilde "),
                 ("^", B + "textasciicircum ")]:
        s = s.replace(a, b)
    return s.strip()


def short(answer):
    """Keep the first sentence. Some reference answers run to three."""
    parts = re.split(r"(?<=[.!?])\s+", answer.strip())
    head, cut = parts[0], len(parts) > 1
    if len(head) > 62 and ", " in head:
        head, cut = head.split(", ", 1)[0], True
    return tex(head.rstrip(".")) + ("." if not cut
                                    else B + "dots")


def frame(img, w, colour, edge):
    img = img.convert("RGB")
    img = img.resize((w, max(1, round(img.height * w / img.width))),
                     Image.LANCZOS)
    img = ImageOps.expand(img, border=1, fill="#FFFFFF")
    return ImageOps.expand(img, border=edge, fill=colour)


records = {r["task_id"]: r
           for r in json.loads(LIC.read_text(encoding="utf-8"))}


def prepare(task_id):
    """Cut the tiles for one task and return everything the tables need."""
    r = records[task_id]
    m = r["meta"]
    scene = m["scene_id"]
    subset, num = scene.split("_", 1)
    ids = sorted(m["candidate_layer_ids"])
    gold = set(m["gold_layer_ids"])
    bg = m["background_layer_id"]

    d = GAL / scene
    d.mkdir(parents=True, exist_ok=True)
    frame(Image.open(SCENES / subset / num / "full.png"), SCENE_W, EDGE, 2
          ).save(d / "scene.png")
    for k in ids:
        frame(Image.open(RENDERS / scene / f"layer_{k}.png"), TILE_W,
              GOLD if k in gold else GREY, 3).save(d / f"layer_{k}.png")

    layers = []
    for k in ids:
        label = "background" if k == bg else m["layer_labels"].get(str(k), "")
        layers.append({"id": k, "label": label, "gold": k in gold,
                       "file": f"gallery/{scene}/layer_{k}.png"})

    shards = []
    for s in r["shards"]:
        used = []
        for im in s.get("images", []):
            hit = re.search(r"layer (\d+)", im["label"])
            if hit:
                used.append(int(hit.group(1)))
        shards.append({"n": s["shard_id"], "text": s["shard"], "layers": used})

    return {"task_id": task_id, "scene": scene, "category": m["category"],
            "question": r["question"], "answer": r["answer"],
            "scene_file": f"gallery/{scene}/scene.png",
            "layers": layers, "shards": shards, "bg": bg}


def chips(used, bg):
    if not used:
        return ""
    out = []
    for k in used:
        out.append(B + "lvqachip{" + ("BG" if k == bg else f"L{k}") + "}")
    return B + "hfill " + "".join(out)


def tiles(e, width):
    """The layer strip: one row of images, one row of captions."""
    n = len(e["layers"])
    L = [B + "setlength{" + B + "lvqatile}{" + B + "dimexpr("
         + width + "-" + str(2 * (n - 1)) + "pt)/" + str(n) + B + "relax}",
         B + "begin{tabular}{@{}" + ("c@{" + B + "hspace{2pt}}") * (n - 1)
         + "c@{}}"]
    L.append(" & ".join(B + "includegraphics[width=" + B + "lvqatile]{"
                        + x["file"] + "}" for x in e["layers"])
             + " " + B + B + "[1pt]")
    caps = []
    for x in e["layers"]:
        tag = "BG" if x["id"] == e["bg"] else f"L{x['id']}"
        colour = "lvqaUp" if x["gold"] else "black!45"
        caps.append(B + "parbox[t]{" + B + "lvqatile}{" + B + "centering"
                    + B + "tiny" + B + "sffamily " + B + "textcolor{" + colour
                    + "}{" + tag + "}" + B + B + "[-1pt] " + tex(x["label"])
                    + "}")
    L.append(" & ".join(caps) + " " + B + B)
    L.append(B + "end{tabular}")
    return L


def shard_block(e, width):
    L = [B + "begin{tabular}{@{}r@{" + B + ";}p{" + width + "}@{}}"]
    for s in e["shards"]:
        L.append(B + "scriptsize " + str(s["n"]) + " & " + B + "scriptsize "
                 + tex(s["text"]) + chips(s["layers"], e["bg"]) + " "
                 + B + B + "[1pt]")
    L.append(B + "end{tabular}")
    return L


def header(e):
    """Category on the left, scene depth and shard count on the right."""
    n = len(e["layers"])
    return (B + "makebox[" + B + "linewidth][s]{" + B + "scriptsize"
            + B + "sffamily" + B + "textcolor{lvqaUp}{"
            + CATNAME[e["category"]].upper() + "}" + B + "hfill"
            + B + "textcolor{black!45}{" + str(n) + " layers "
            + B + "quad " + str(len(e["shards"])) + " shards}}")


def pane(width, body):
    """A tabular cell that behaves like a page. Inside a minipage a line
    break is a line break, where inside a p-column it ends the whole row."""
    return ([B + "begin{minipage}[t]{" + width + "}" + B + "vspace{0pt}"]
            + body + [B + "end{minipage}"])


def left(e):
    return [B + "includegraphics[width=" + B + "linewidth]{"
            + e["scene_file"] + "}" + B + B + "[3pt]"] + tiles(e, B
                                                               + "linewidth")


def right(e):
    return ([B + "raggedright", header(e) + B + B + "[5pt]",
             "{" + B + "small" + B + "itshape " + tex(e["question"]) + "}"
             + B + B + "[4pt]",
             "{" + B + "footnotesize" + B + "textcolor{black!50}{Answer}"
             + B + "quad " + B + "textbf{" + short(e["answer"]) + "}}"
             + B + B + "[6pt]"]
            + shard_block(e, B + "dimexpr" + B + "linewidth-12pt" + B
                          + "relax"))


# ----------------------------------------------------------------- main figure
LW, RW = "0.415" + B + "textwidth", "0.545" + B + "textwidth"
main = [prepare(t) for t in MAIN]
F = ["% Main-paper figure. Needs: graphicx, booktabs, xcolor, array.",
     "% Add these two lines to the preamble once:",
     "%   " + B + "newlength{" + B + "lvqatile}",
     "%   " + B + "newcommand{" + B + "lvqachip}[1]{{" + B
     + "setlength{" + B + "fboxsep}{1.6pt}" + B + "colorbox{lvqaUp!14}{"
     + B + "scriptsize" + B + "sffamily" + B
     + "textcolor{lvqaUp!75!black}{#1}}}}",
     B + "begin{figure}[t]", B + "centering",
     B + "setlength{" + B + "tabcolsep}{0pt}",
     B + "begin{tabular}{@{}l@{" + B + "hspace{14pt}}l@{}}", B + "toprule"]
for i, e in enumerate(main):
    if i:
        F.append(B + "midrule")
    F += pane(LW, left(e)) + ["&"] + pane(RW, right(e)) + [B + B + "[2pt]"]
F += [B + "bottomrule", B + "end{tabular}",
      B + "caption{" + B + "textbf{One question, one scene, and the parts each "
      "is broken into.}",
      "Left: the composed scene, then every layer it is built from.",
      "A teal border marks a layer the question depends on, grey marks one it "
      "does not.",
      "Right: the question, the reference answer, and the shards the question "
      "is delivered as.",
      "A chip on a shard names the layer that arrives with it under "
      + B + "textsc{aligned}.",
      "The three rows are an occlusion that exists only in the composite, a "
      "count whose members sit one per layer, and a chain of references that "
      "needs four layers at once.}",
      B + "label{fig:examples}", B + "end{figure}"]
(OUT / "fig_examples.tex").write_text("\n".join(F) + "\n", encoding="utf-8")

# -------------------------------------------------------------- appendix table
DW, IW, TW = ("0.055" + B + "textwidth", "0.345" + B + "textwidth",
              "0.520" + B + "textwidth")
scale = [prepare(t) for t in SCALE]
T = ["% Appendix table. Needs: graphicx, booktabs, xcolor, array, longtable.",
     "% Uses " + B + "lvqatile and " + B + "lvqachip from fig_examples.tex.",
     "% A longtable rather than a table, because five worked examples do not",
     "% fit on one page and a row must never break in the middle.",
     "{" + B + "footnotesize",
     B + "setlength{" + B + "tabcolsep}{0pt}",
     B + "setlength{" + B + "LTcapwidth}{" + B + "textwidth}",
     B + "begin{longtable}{@{}l@{" + B + "hspace{8pt}}l@{" + B
     + "hspace{12pt}}l@{}}",
     B + "caption{" + B + "textbf{Scene depth across the benchmark.}",
     "One scene per depth, from a single object over a background up to five.",
     "Borders and chips read as in Figure~" + B + "ref{fig:examples}.",
     "Depth is a property of the scene and shard count is a property of the "
     "question, so the two do not move together.}",
     B + "label{tab:layer-scale}" + B + B, B + "toprule", B + "endfirsthead",
     B + "multicolumn{3}{@{}l}{" + B + "footnotesize" + B + "itshape Table "
     + B + "thetable, continued from the previous page.}" + B + B,
     B + "toprule", B + "endhead",
     B + "bottomrule", B + "endlastfoot"]
for i, e in enumerate(scale):
    if i:
        T.append(B + "midrule")
    n = len(e["layers"])
    depth = [B + "centering" + B + "vspace{2pt}", B + "large" + B + "textbf{"
             + str(n) + "}" + B + B + "[-3pt]",
             "{" + B + "tiny" + B + "sffamily" + B + "textcolor{black!50}{"
             + str(n - 1) + " + bg}}"]
    T += (pane(DW, depth) + ["&"] + pane(IW, left(e)) + ["&"]
          + pane(TW, right(e)) + [B + B + "[2pt]"])
T += [B + "end{longtable}", "}"]
(OUT / "tab_layer_scale.tex").write_text("\n".join(T) + "\n", encoding="utf-8")

manifest = {"main": main, "scale": scale}
(GAL / "examples.json").write_text(json.dumps(manifest, indent=2),
                                   encoding="utf-8")

print("wrote fig_examples.tex and tab_layer_scale.tex")
for e in main + scale:
    print(f"  {e['task_id']:44} {len(e['layers'])} layers  "
          f"{len(e['shards'])} shards")
print(f"  tiles in {GAL.relative_to(ROOT)}: "
      f"{sum(1 for _ in GAL.rglob('*.png'))} images")
