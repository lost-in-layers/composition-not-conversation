"""Draw the conversations that people will grade, to validate the consensus judge.

The draw is stratified by condition, so all sixteen are represented equally,
and models are spread by cycling through them inside each condition.

There is no oversampling of hard cases. The eleven judges split on 72% of
conversations and agree unanimously on only 28%, so disagreement is already the
common case and a uniform draw lands plenty of it. Each item still records
which stratum it came from, so agreement can be reported separately for the
unanimous and the split halves.

Usage:  python sample_judge_validation.py [--n 400] [--out human_validation]
Run from vendor/conv-smi (where logs/ lives).
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

LOGS = Path("logs")
RUNS = ["Final-0", "Final-1", "Final-2"]

COND = {
    ("full", "layered_vqa_text.json"): "FULL",
    ("concat", "layered_vqa_text.json"): "CONCAT-Q",
    ("sharded", "layered_vqa_text.json"): "SHARDED-Q",
    ("recap-concat", "layered_vqa_text.json"): "RECAP-Q",
    ("snowball", "layered_vqa_text.json"): "SNOWBALL-Q",
    ("full", "layered_vqa_visual_canonical.json"): "CONCAT-V",
    ("sharded-scripted", "layered_vqa_visual_canonical.json"): "SHARDED-V",
    ("recap-concat", "layered_vqa_visual_canonical.json"): "RECAP-V",
    ("snowball-scripted", "layered_vqa_visual_canonical.json"): "SNOWBALL-V",
    ("sharded-scripted", "layered_vqa_aligned_vq.json"): "ALIGNED-VQ",
    ("sharded-scripted", "layered_vqa_misaligned_vq.json"): "MISALIGNED-VQ",
    ("sharded-scripted", "layered_vqa_visual_evidence_first.json"): "EVIDENCE-FIRST",
    ("sharded-scripted", "layered_vqa_visual_evidence_last.json"): "EVIDENCE-LAST",
    ("shuffle-concat", "layered_vqa_text.json"): "SHUFFLED-CONCAT",
    ("full", "layered_vqa_grounding.json"): "ALL-LAYERS",
    ("full", "layered_vqa_oracle_gold.json"): "ORACLE",
}
DATASETS = Path("../../data/derived/lic")


def scan_votes(cache):
    """conv_id -> judge verdicts, read once and cached.

    The post-hoc tree is 1.7M records, so a rerun of the sampler should not
    have to read it again.
    """
    if cache.exists():
        print(f"votes from cache: {cache}")
        return json.loads(cache.read_text(encoding="utf-8"))

    votes = {}
    seen = 0
    for run in RUNS:
        root = LOGS / run / "posthoc"
        if not root.is_dir():
            continue
        for jdir in sorted(p for p in root.iterdir() if p.is_dir()):
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
                    seen += 1
                    cond = COND.get((rec.get("conv_type"), rec.get("dataset_fn")))
                    cid = rec.get("conv_id")
                    if cond is None or not cid:
                        continue
                    entry = votes.get(cid)
                    if entry is None:
                        entry = votes[cid] = {
                            "conv_id": cid, "run": run, "cond": cond,
                            "task_id": rec.get("task_id"),
                            "model": rec.get("assistant_model"),
                            "yes": 0, "n": 0,
                        }
                    entry["n"] += 1
                    entry["yes"] += 1 if rec.get("final_turn_correct") else 0
            print(f"  {run}/{jdir.name}: {seen:,} records", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(votes), encoding="utf-8")
    print(f"cached votes for {len(votes):,} conversations")
    return votes


def stratified(pool, want, rng):
    """Take `want` items: an equal quota per condition, models round-robin.

    The quota is per condition rather than global, so no condition can crowd
    another out. Inside a condition the draw cycles through models, which
    spreads the sample over all eleven without a second explicit stratum.
    """
    by_cond = defaultdict(lambda: defaultdict(list))
    for item in pool:
        by_cond[item["cond"]][item["model"]].append(item)
    for cond in by_cond:
        for model in by_cond[cond]:
            rng.shuffle(by_cond[cond][model])

    conds = sorted(by_cond)
    quota = want // max(1, len(conds))
    picked = []
    for cond in conds:
        models = sorted(by_cond[cond])
        taken, m = 0, 0
        while taken < quota and any(by_cond[cond][x] for x in models):
            bucket = by_cond[cond][models[m % len(models)]]
            if bucket:
                picked.append(bucket.pop())
                taken += 1
            m += 1
    return picked


def load_answers():
    answers, questions, categories = {}, {}, {}
    for path in DATASETS.glob("layered_vqa_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for row in data if isinstance(data, list) else []:
            tid = row.get("task_id")
            if tid and tid not in answers:
                answers[tid] = row.get("answer")
                questions[tid] = row.get("question")
                categories[tid] = (row.get("meta") or {}).get("category")
    return answers, questions, categories


def normalise(trace):
    """Flatten a trace into text and repo-relative image paths.

    Image URLs in a log are relative to the dataset JSON's own directory, so
    they mean nothing to a web page. Resolving them here keeps the review app
    from having to know where the datasets live.
    """
    out = []
    for message in trace or []:
        # `log` entries hold the judge's own verdict and reasoning, and the
        # system turn is the same prompt in every conversation. Neither belongs
        # in front of a reviewer, and the verdict must never reach the page.
        if message.get("role") not in ("user", "assistant"):
            continue
        parts = []
        content = message.get("content")
        if isinstance(content, str):
            parts.append({"type": "text", "text": content})
        for part in content if isinstance(content, list) else []:
            if part.get("type") == "text":
                parts.append({"type": "text", "text": part.get("text", "")})
            elif part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                try:
                    full = (DATASETS / url).resolve()
                    rel = full.relative_to(DATASETS.parents[2].resolve()).as_posix()
                except Exception:
                    rel = None
                parts.append({"type": "image", "path": rel,
                              "label": part.get("label") or Path(url).stem})
        out.append({"role": message.get("role", "user"), "parts": parts})
    return out


def find_traces(wanted):
    """One pass over the primary logs, pulling only the chosen conversations."""
    found = {}
    for run in RUNS:
        root = LOGS / run / "layered_vqa"
        if not root.is_dir():
            continue
        for path in root.rglob("*.jsonl"):
            for line in path.open(encoding="utf-8", errors="replace"):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                cid = rec.get("conv_id")
                if cid in wanted and cid not in found:
                    found[cid] = rec.get("trace") or []
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default="human_validation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    votes = scan_votes(out / "votes_cache.json")
    rng = random.Random(args.seed)

    full = [v for v in votes.values() if v["n"] >= 11]
    split = [v for v in full if 0 < v["yes"] < v["n"]]
    unanimous = [v for v in full if v["yes"] in (0, v["n"])]
    print(f"conversations with 11 votes: {len(full):,}")
    print(f"  unanimous {len(unanimous):,} ({len(unanimous)/len(full):.1%})")
    print(f"  split     {len(split):,} ({len(split)/len(full):.1%})")

    # Disagreement is the common case here, not the rare one, so the sample is
    # a plain stratified draw over all conversations. The unanimous/split label
    # is still recorded, so agreement can be reported separately for each.
    chosen = stratified(full, args.n, rng)
    rng.shuffle(chosen)

    answers, questions, categories = load_answers()
    print("reading transcripts...")
    traces = find_traces({c["conv_id"] for c in chosen})

    items = []
    for i, c in enumerate(chosen, 1):
        tid = c["task_id"]
        items.append({
            "index": i,
            "conv_id": c["conv_id"],
            "run": c["run"],
            "condition": c["cond"],
            "model": c["model"],
            "task_id": tid,
            "category": categories.get(tid),
            "question": questions.get(tid),
            "reference_answer": answers.get(tid),
            "stratum": "unanimous" if c["yes"] in (0, c["n"]) else "split",
            "trace": normalise(traces.get(c["conv_id"], [])),
            # Hidden from the app. Kept here so scoring needs one file.
            "_judges": {"yes": c["yes"], "n": c["n"]},
        })

    missing = sum(1 for it in items if not it["trace"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "sample.json").write_text(
        json.dumps({"seed": args.seed, "n": len(items), "items": items},
                   indent=1), encoding="utf-8")
    print(f"wrote {out/'sample.json'}: {len(items)} conversations, "
          f"{missing} without a transcript")
    by = defaultdict(int)
    for it in items:
        by[it["condition"]] += 1
    print("per condition:", dict(sorted(by.items())))


if __name__ == "__main__":
    main()
