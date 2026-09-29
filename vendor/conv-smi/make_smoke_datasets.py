#!/usr/bin/env python
"""Build tiny copies of the eight Layered-VQA dataset files for a smoke test.

A smoke run wants every delivery condition to execute end to end, not to
execute on all 300 samples. There is no `--limit` flag on the simulators, and
adding one would be a second way for a real run to silently score a subset, so
the size is cut where it belongs: in the data. Each source file is truncated to
the first N samples and written under `<dataset_dir>/.smoke-n<N>/` with its
**original basename** — `score_experiment_a/b` and the logs both key conditions
off that basename, so renaming the files would make the smoke logs unscorable.

Image references are relative to the dataset JSON's own directory
(`multimodal.image_base_dir`), and the smoke directory sits one level deeper,
so every relative path gains a `../`. Absolute paths and URLs are left alone.

The same scenes are chosen in every file where they exist, so the conditions
compare like for like; a file that shares no scene id with the text dataset
falls back to its own first N samples rather than coming out empty.

The output directory carries a self-ignoring `.gitignore`, because
`data/derived/lic/` is tracked and a smoke run should not show up in
`git status`.

    python make_smoke_datasets.py --n 1
    python make_smoke_datasets.py --n 3 --dataset_dir ../../data/derived/lic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# The eight files the 14 Experiment-B conditions and the 3 Experiment-A
# conditions read between them.
DATASETS = (
    "layered_vqa_text.json",
    "layered_vqa_visual_canonical.json",
    "layered_vqa_grounding.json",
    "layered_vqa_oracle_gold.json",
    "layered_vqa_aligned_vq.json",
    "layered_vqa_misaligned_vq.json",
    "layered_vqa_visual_evidence_first.json",
    "layered_vqa_visual_evidence_last.json",
)

IMAGE_KEYS = ("images", "recap_images")


def _is_relative_local(path: str) -> bool:
    if not isinstance(path, str) or not path:
        return False
    if "://" in path or path.startswith("data:"):
        return False
    candidate = Path(path)
    if candidate.is_absolute():
        return False
    # Windows drive letters ("C:/...") are absolute to multimodal too.
    return not (len(path) > 1 and path[1] == ":")


def _rewrite(value, prefix: str):
    """Prefix every relative image path in `value`, in place where possible."""
    if isinstance(value, str):
        return prefix + value if _is_relative_local(value) else value
    if isinstance(value, list):
        return [_rewrite(item, prefix) for item in value]
    if isinstance(value, dict):
        out = dict(value)
        if isinstance(out.get("path"), str):
            out["path"] = _rewrite(out["path"], prefix)
        for key in IMAGE_KEYS:
            if key in out:
                out[key] = _rewrite(out[key], prefix)
        if "shards" in out:
            out["shards"] = _rewrite(out["shards"], prefix)
        return out
    return value


def scene_id(sample: dict) -> str | None:
    meta = sample.get("meta") or {}
    return meta.get("scene_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, required=True, help="Samples per dataset file")
    parser.add_argument("--dataset_dir", default="../../data/derived/lic")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Defaults to <dataset_dir>/.smoke-n<N>. Must stay exactly one "
        "directory below dataset_dir, or the rewritten image paths miss.",
    )
    args = parser.parse_args()

    if args.n < 1:
        raise SystemExit("--n must be at least 1")

    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else dataset_dir / f".smoke-n{args.n}"
    if out_dir.parent != dataset_dir:
        raise SystemExit(
            f"{out_dir} is not directly inside {dataset_dir}; image paths would "
            f"be rewritten by the wrong number of levels."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    # Self-ignoring: data/derived/lic/ is tracked, and a smoke run is not data.
    (out_dir / ".gitignore").write_text("*\n", encoding="utf-8")

    anchor = json.loads((dataset_dir / DATASETS[0]).read_text(encoding="utf-8"))
    wanted: list[str] = []
    for sample in anchor:
        sid = scene_id(sample)
        if sid is not None and sid not in wanted:
            wanted.append(sid)
        if len(wanted) >= args.n:
            break
    wanted_set = set(wanted)

    for name in DATASETS:
        source = dataset_dir / name
        if not source.exists():
            raise SystemExit(f"dataset not found: {source}")
        samples = json.loads(source.read_text(encoding="utf-8"))

        picked = [s for s in samples if scene_id(s) in wanted_set][: args.n]
        if not picked:
            # No shared scenes with the text dataset: still exercise the
            # condition rather than writing an empty file it would skip.
            picked = samples[: args.n]

        payload = [_rewrite(sample, "../") for sample in picked]
        (out_dir / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"{name}: {len(payload)} sample(s)")

    print(f"smoke datasets: {out_dir}")


if __name__ == "__main__":
    main()
