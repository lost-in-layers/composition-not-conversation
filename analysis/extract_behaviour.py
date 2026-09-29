"""One pass over the post-hoc matrix, producing the compact CSVs that the
figure and table generators read.

Two kinds of field live on a post-hoc record and they are pooled differently:

  * Judge-independent -- response_chars, shards_total, layer_mentions,
    layer_intro_turn, gold_layer_ids. These are read off the conversation log,
    so all eleven judges report the same values; the first one seen is kept.

  * Judge-dependent -- n_attempts, first_attempt_progress, attempt_lengths.
    These come from the judge's reading of what counts as an answer attempt,
    and judges genuinely disagree. Averaging over all eleven is more stable
    than trusting any single one, so these accumulate across judges.

Correctness is always the consensus final-turn verdict (majority vote across
judges), matching the two main tables.

Usage:  python overleaf/extract_behaviour.py <out-dir>
Run from vendor/conv-smi (where logs/ lives).
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
LOGS = Path("logs")
RUNS = ["Final-0", "Final-1", "Final-2"]
CLOSED_RUNS = ["Final-Closed-0", "Final-Closed-1", "Final-Closed-2"]

# (conv_type, dataset_fn) -> (condition key, family)
COND = {
    ("full", "layered_vqa_text.json"): ("B1", "Text"),
    ("concat", "layered_vqa_text.json"): ("B2", "Text"),
    ("sharded", "layered_vqa_text.json"): ("B3", "Text"),
    ("recap-concat", "layered_vqa_text.json"): ("B4", "Text"),
    ("snowball", "layered_vqa_text.json"): ("B5", "Text"),
    ("full", "layered_vqa_visual_canonical.json"): ("B6", "Visual"),
    ("sharded-scripted", "layered_vqa_visual_canonical.json"): ("B7", "Visual"),
    ("recap-concat", "layered_vqa_visual_canonical.json"): ("B8", "Visual"),
    ("snowball-scripted", "layered_vqa_visual_canonical.json"): ("B9", "Visual"),
    ("sharded-scripted", "layered_vqa_aligned_vq.json"): ("B10", "Layered"),
    ("sharded-scripted", "layered_vqa_misaligned_vq.json"): ("B11", "Layered"),
    ("sharded-scripted", "layered_vqa_visual_evidence_first.json"): ("B12", "Order"),
    ("sharded-scripted", "layered_vqa_visual_evidence_last.json"): ("B13", "Order"),
    ("shuffle-concat", "layered_vqa_text.json"): ("B14", "Order"),
    ("full", "layered_vqa_grounding.json"): ("A2", "ExpA"),
    ("full", "layered_vqa_oracle_gold.json"): ("A3", "ExpA"),
}
# The multi-turn conditions -- the ones where "when did it answer" is a question
MULTI = {"B3", "B4", "B5", "B7", "B8", "B9", "B10", "B11", "B12", "B13"}
SINGLE = {"B1", "B2", "B6", "B14"}


def bin_of(progress):
    """First-attempt position as a fifth of the conversation."""
    if progress is None:
        return None
    for hi, lab in ((0.2, "0-20"), (0.4, "20-40"), (0.6, "40-60"), (0.8, "60-80")):
        if progress <= hi:
            return lab
    return "80-100"


conv = {}                                  # key -> per-conversation accumulator
bloat = defaultdict(lambda: [0, 0.0, 0])   # (family, attempt_idx) -> [n, chars, correct]
ref_len = defaultdict(lambda: [0, 0.0])    # (family, cond) -> [n, chars] for the single-turn refs
cite = defaultdict(int)                    # (intro_turn, mention_turn, role) -> count
selfjudge = defaultdict(lambda: [0, 0])    # (judge, assistant) -> [n, correct]

def sources():
    """Every directory of records, and whether it was cross-judged.

    The open models were judged by all eleven, one directory per judge, and a
    conversation's verdict is the majority. The closed models were judged once
    by the run's own judge, because cross-judging three closed models against
    eleven others would cost more than it would tell us. A majority of one
    vote is that vote, so the same accumulator serves both. What the closed
    models cannot join is the cross-judge work: self-preference, the
    question-level bootstrap, and the answer-bloat curves stay open-only.
    """
    for run in RUNS:
        root = LOGS / run / "posthoc"
        if root.is_dir():
            for jdir in sorted(root.iterdir()):
                if jdir.is_dir():
                    yield run, jdir.name, jdir, True
    for run in CLOSED_RUNS:
        root = LOGS / run / "primary-analysis"
        if root.is_dir():
            for mdir in sorted(root.iterdir()):
                if mdir.is_dir():
                    yield run, "primary", mdir, False


n_records = 0
for run, judge, jdir, cross in sources():
    for path in jdir.rglob("*.jsonl"):
        if path.name in ("judge_cache.jsonl", "errors.jsonl"):
            continue
        for line in path.open(encoding="utf-8", errors="replace"):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            n_records += 1
            spec = (rec.get("conv_type"), rec.get("dataset_fn"))
            if spec not in COND:
                continue
            cond, family = COND[spec]
            model = rec.get("assistant_model")
            key = (run, model, cond, rec.get("task_id"))

            acc = conv.get(key)
            if acc is None:
                acc = conv[key] = {
                    "votes": 0, "correct": 0, "prog": 0.0, "nprog": 0,
                    "att": 0.0, "natt": 0, "ungr": 0,
                    "chars": rec.get("mean_response_chars") or 0.0,
                    "total_chars": rec.get("total_chars") or 0,
                    "shards": rec.get("shards_total") or 0,
                    "family": family, "cross": cross,
                }
                # Judge-independent, so recorded once: which layer was
                # mentioned in which turn, against when it was introduced.
                intro = rec.get("layer_intro_turn") or {}
                gold = set(rec.get("gold_layer_ids") or [])
                for turn_idx, mentions in enumerate(rec.get("layer_mentions") or [], 1):
                    for lid in mentions:
                        t0 = intro.get(str(lid))
                        if t0 and cross:
                            role = "gold" if lid in gold else "distractor"
                            cite[(t0, turn_idx, role)] += 1

            acc["votes"] += 1
            acc["correct"] += 1 if rec.get("final_turn_correct") else 0
            acc["ungr"] += 1 if rec.get("ungrounded_correct") else 0
            p = rec.get("first_attempt_progress")
            if p is not None:
                acc["prog"] += p
                acc["nprog"] += 1
            acc["att"] += rec.get("n_attempts") or 0
            acc["natt"] += 1

            if cross:
                sj = selfjudge[(judge, model)]
                sj[0] += 1
                sj[1] += 1 if rec.get("final_turn_correct") else 0

            for item in (rec.get("attempt_lengths") or []) if cross else []:
                idx = item.get("attempt_index")
                if idx and idx <= 12:
                    b = bloat[(family, cond in SINGLE, idx)]
                    b[0] += 1
                    b[1] += item.get("chars") or 0
                    b[2] += 1 if item.get("correct") else 0

# --- roll conversations up ---------------------------------------------------
per_cond = defaultdict(lambda: [0, 0, 0, 0])       # (model,cond) -> n, correct, ungrounded, both
per_shard = defaultdict(lambda: [0, 0])            # (model,family,shards) -> n, correct
per_bin = defaultdict(lambda: [0, 0])              # (model,bin) -> n, correct
per_q = defaultdict(lambda: [0, 0])                # (model,cond,task) -> n, correct  (for bootstrap)
verb_rows = defaultdict(list)                      # (model,cond) -> [(chars, correct)]

for (run, model, cond, task), a in conv.items():
    correct = 1 if a["correct"] * 2 > a["votes"] else 0
    ungrounded = 1 if a["ungr"] * 2 > a["votes"] else 0
    c = per_cond[(model, cond)]
    c[0] += 1
    c[1] += correct
    c[2] += ungrounded
    # An ungrounded answer is not automatically an answer we score. A model can
    # produce the right answer early, before its evidence arrived, and then talk
    # itself out of it: about one in six ungrounded conversations ends wrong.
    # Only the conjunction is a share of the answers the paper counts.
    c[3] += ungrounded * correct
    if a["cross"]:
        # The bootstrap and the category table are cross-judge results, so the
        # closed models stay out of the question-level file.
        q = per_q[(model, cond, task)]
        q[0] += 1
        q[1] += correct
    if cond in MULTI:
        s = per_shard[(model, a["family"], a["shards"])]
        s[0] += 1
        s[1] += correct
        if a["nprog"]:
            b = per_bin[(model, bin_of(a["prog"] / a["nprog"]))]
            b[0] += 1
            b[1] += correct
    verb_rows[(model, cond)].append((a["chars"], correct))

# Verbosity quintiles are computed inside each (model, condition) so the bins
# compare a model against its own length distribution, not against a chatty
# model's. Pooling first would just rediscover which models are verbose.
per_verb = defaultdict(lambda: [0, 0])
for (model, cond), rows in verb_rows.items():
    rows.sort(key=lambda r: r[0])
    n = len(rows)
    if n < 5:
        continue
    fam = "single" if cond in SINGLE else "multi"
    for i, (_, correct) in enumerate(rows):
        q = min(4, (i * 5) // n)
        v = per_verb[(model, fam, q)]
        v[0] += 1
        v[1] += correct


def dump(name, header, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  {name}: {len(rows)} rows")


print(f"records read: {n_records:,}   conversations: {len(conv):,}")
dump("beh_cond.csv", ["model", "cond", "n", "correct", "ungrounded", "both"],
     [[m, c, *v] for (m, c), v in sorted(per_cond.items())])
dump("beh_shards.csv", ["model", "family", "shards", "n", "correct"],
     [[m, f, s, *v] for (m, f, s), v in sorted(per_shard.items())])
dump("beh_bins.csv", ["model", "bin", "n", "correct"],
     [[m, b, *v] for (m, b), v in sorted(per_bin.items()) if b])
dump("beh_verbosity.csv", ["model", "family", "quintile", "n", "correct"],
     [[m, f, q, *v] for (m, f, q), v in sorted(per_verb.items())])
dump("beh_bloat.csv", ["family", "single_turn", "attempt", "n", "chars", "correct"],
     [[f, int(s), i, *v] for (f, s, i), v in sorted(bloat.items())])
dump("beh_citation.csv", ["intro_turn", "mention_turn", "role", "count"],
     [[a, b, c, v] for (a, b, c), v in sorted(cite.items())])
dump("beh_judge.csv", ["judge", "assistant", "n", "correct"],
     [[j, a, *v] for (j, a), v in sorted(selfjudge.items())])
dump("beh_question.csv", ["model", "cond", "task_id", "runs", "correct"],
     [[m, c, t, *v] for (m, c, t), v in sorted(per_q.items())])
