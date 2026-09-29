"""Deliver the gold layers the misaligned export dropped, without un-misaligning it.

MISALIGNED_VQ delivers each question shard together with layers that do *not*
support it. On 15 of the 300 questions every shard supports the gold group
(`question_shard_support` lists it under all of them), so no placement of that
group is misaligned. The exporter's fallback was to drop the group and send the
background layer in its place: the turn stayed misaligned, but the gold layers
then arrived on no turn at all, and the model was answering blind. Those samples
carry a `meta.misaligned_fallback` marker recording the substitution.

The fix is *not* to put the gold group back on a shard it supports — that turns
the sample into a delayed ALIGNED_VQ and makes the condition measure the
opposite of what it is for. Instead:

  * the background-layer delivery stays exactly where it is, so every shard the
    export already scheduled still arrives with a non-supporting layer, and
  * the dropped gold group is added on **one** further shard, as late as the
    schedule allows, so it is still available by the end of the conversation.

The make-up shard is chosen as the last shard that carries no images yet and,
where such a shard exists, supports none of the added layers (only
`lvis_000000500507::counting` has one). Landing the group at the end is the
strongest separation still available: ALIGNED_VQ hands these layers over on the
first shard, so the two arms differ by the whole conversation.

`meta.misaligned_violations` stays empty on all 300 samples, so no sample is
excluded from the aligned-vs-misaligned contrast. What was done instead is
recorded per sample in `meta.misaligned_gold_makeup`, which replaces the old
`misaligned_fallback` marker.

Rewrites, for the 15 affected samples only:

    shards[*].images            adds the gold group to the make-up shard
    meta.misaligned_schedule    the schedule actually delivered
    meta.reveal_order           the order layers arrive in
    meta.misaligned_gold_makeup what was added, where, and why

Usage:

    python src/vecgen/repair_misaligned_vq_schedule.py            # rewrite in place
    python src/vecgen/repair_misaligned_vq_schedule.py --check    # report only
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path("data/derived/lic/layered_vqa_misaligned_vq.json")
LAYER_RE = re.compile(r"layer_(\d+)\.png$")


def layer_ids_of(shard: dict[str, Any]) -> list[int]:
    ids = []
    for image in shard.get("images") or []:
        match = LAYER_RE.search(image["path"])
        if match:
            ids.append(int(match.group(1)))
    return ids


def image_ref(sample: dict[str, Any], layer_id: int, detail: str) -> dict[str, str]:
    """The same ref `export_lic_dataset.layer_image_refs` writes."""
    label = f"layer {layer_id}"
    if sample["meta"].get("background_layer_id") == layer_id:
        label = f"layer {layer_id} (the background layer)"
    return {
        "path": f"../lic_renders/{sample['meta']['scene_id']}/layer_{layer_id}.png",
        "detail": detail,
        "label": label,
    }


def detail_of(sample: dict[str, Any]) -> str:
    for shard in sample["shards"]:
        for image in shard.get("images") or []:
            return image.get("detail", "high")
    for image in sample.get("recap_images") or []:
        return image.get("detail", "high")
    return "high"


def choose_makeup_shard(
    sample: dict[str, Any], missing: list[int]
) -> tuple[int, bool]:
    """The shard to hang the recovered gold group on, and whether it supports it.

    Preference order: a free shard that supports none of the added layers (a
    real misaligned slot), then the last free shard (latest possible arrival),
    then the last shard. "Free" means the export scheduled no image there, so
    the make-up delivery never changes an existing turn's image count.
    """
    support = {
        int(k): {int(x) for x in v}
        for k, v in sample["meta"]["question_shard_support"].items()
    }
    added = set(missing)
    shard_ids = [int(shard["shard_id"]) for shard in sample["shards"]]
    free = [
        int(shard["shard_id"])
        for shard in sample["shards"]
        if not (shard.get("images") or [])
    ]

    unsupported = [sid for sid in free if not (added & support.get(sid, set()))]
    if unsupported:
        return unsupported[-1], False
    if free:
        return free[-1], True
    return shard_ids[-1], bool(added & support.get(shard_ids[-1], set()))


def repair_sample(sample: dict[str, Any], render_root: Path | None) -> dict[str, Any] | None:
    """Add the dropped gold group back; return a report row, or None if untouched."""
    meta = sample["meta"]
    fallback = meta.get("misaligned_fallback")
    if not fallback:
        return None

    delivered = {lid for shard in sample["shards"] for lid in layer_ids_of(shard)}
    missing = sorted(int(x) for x in fallback["removed_layer_ids"] if int(x) not in delivered)
    if not missing:
        return None

    target, supported = choose_makeup_shard(sample, missing)
    detail = detail_of(sample)

    for shard in sample["shards"]:
        if int(shard["shard_id"]) != target:
            continue
        images = list(shard.get("images") or [])
        for layer_id in missing:
            if render_root is not None:
                path = (render_root / meta["scene_id"] / f"layer_{layer_id}.png").resolve()
                if not path.exists():
                    raise SystemExit(
                        f"{sample['task_id']}: render missing for layer {layer_id}: {path}"
                    )
            images.append(image_ref(sample, layer_id, detail))
        shard["images"] = images
        break
    else:
        raise SystemExit(f"{sample['task_id']}: no shard {target}")

    meta["misaligned_schedule"] = {
        str(shard["shard_id"]): layer_ids_of(shard) for shard in sample["shards"]
    }
    meta["reveal_order"] = [
        lid for shard in sample["shards"] for lid in layer_ids_of(shard)
    ]
    # The old marker described a delivery that no longer happens. Replace it with
    # what this schedule actually does, keeping the reason it had to.
    meta.pop("misaligned_fallback", None)
    meta["misaligned_gold_makeup"] = {
        "reason": "every_shard_supports_the_gold_group",
        "substituted_layer_id": fallback.get("fallback_layer_id"),
        "substituted_shard_id": fallback.get("fallback_shard_id"),
        "added_layer_ids": missing,
        "added_shard_id": target,
        "added_shard_supports_them": supported,
    }

    gold = sorted(int(x) for x in meta.get("gold_layer_ids", []))
    still_missing = sorted(
        set(gold) - {lid for shard in sample["shards"] for lid in layer_ids_of(shard)}
    )
    if still_missing:
        raise SystemExit(
            f"{sample['task_id']}: gold layers {still_missing} still undelivered"
        )

    return {
        "task_id": sample["task_id"],
        "gold": gold,
        "added": missing,
        "shard": target,
        "supported": supported,
        "shards": len(sample["shards"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report the affected samples without writing the file",
    )
    parser.add_argument(
        "--render_dir",
        type=Path,
        default=None,
        help="Verify every newly attached render exists under this directory "
        "(default: the dataset's sibling lic_renders/)",
    )
    args = parser.parse_args(argv)

    raw = args.path.read_bytes().decode("utf-8")
    # Match the file as it sits on disk: the export is ASCII-escaped, and the
    # checkout's line endings must survive or every line shows up in the diff.
    newline = "\r\n" if "\r\n" in raw else "\n"
    samples = json.loads(raw)

    render_root = args.render_dir
    if render_root is None:
        render_root = args.path.parent.parent / "lic_renders"
    if not render_root.is_dir():
        print(f"[warn] {render_root} not found; skipping render existence checks")
        render_root = None

    repaired = [row for row in (repair_sample(s, render_root) for s in samples) if row]

    for row in repaired:
        note = "on a shard that supports them" if row["supported"] else "on a non-supporting shard"
        print(
            f"{row['task_id']}: gold={row['gold']} added={row['added']} "
            f"-> shard {row['shard']}/{row['shards']} ({note})"
        )
    print(f"\n{len(repaired)}/{len(samples)} samples repaired")

    if args.check:
        print("--check: file not written")
        return 1 if repaired else 0

    if repaired:
        text = json.dumps(samples, indent=2) + "\n"
        args.path.write_bytes(text.replace("\n", newline).encode("utf-8"))
        print(f"wrote {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
