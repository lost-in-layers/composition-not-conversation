from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm


LAYER_PATTERN = re.compile(r"^layer_(\d+)\.png$", re.IGNORECASE)


def layer_number(path: Path) -> int:
    match = LAYER_PATTERN.match(path.name)
    if not match:
        raise ValueError(f"Not a layer file: {path}")
    return int(match.group(1))


def make_strip(steps: list[Image.Image], gap: int = 12) -> Image.Image:
    height = max(image.height for image in steps)
    width = sum(image.width for image in steps) + gap * (len(steps) + 1)

    strip = Image.new("RGB", (width, height + 2 * gap), (20, 20, 20))
    x = gap
    for image in steps:
        strip.paste(image, (x, gap + (height - image.height) // 2))
        x += image.width + gap
    return strip


def process_directory(
    directory_text: str,
    gap: int,
    overwrite: bool,
) -> tuple[str, str]:
    directory = Path(directory_text)
    output = directory / "progressive_strip.png"

    if output.exists() and not overwrite:
        return "existing", str(output)

    layer_paths = sorted(
        (
            path
            for path in directory.glob("layer_*.png")
            if LAYER_PATTERN.match(path.name)
        ),
        key=layer_number,
    )

    if not layer_paths or layer_number(layer_paths[0]) != 0:
        return "failed", f"{directory}: layer_0.png missing"

    with Image.open(layer_paths[0]) as background_image:
        canvas = background_image.convert("RGBA")

    steps = [canvas.convert("RGB")]

    for layer_path in layer_paths[1:]:
        with Image.open(layer_path) as layer_image:
            layer = layer_image.convert("RGBA")
        canvas = Image.alpha_composite(canvas, layer)
        steps.append(canvas.convert("RGB"))

    make_strip(steps, gap=gap).save(output)
    return "created", str(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create progressive_strip.png directly from existing MuLAn "
            "layer_*.png renders. No .p.zl annotations are required."
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        help=r"Root render directory, such as .\renders\lvis or .\renders\laion.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gap", type=int, default=12)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Render root not found: {root}")

    directories = sorted({
        path.parent
        for path in root.rglob("layer_0.png")
        if (path.parent / "full.png").exists()
    })

    if args.limit:
        directories = directories[: args.limit]

    print(f"Complete rendered image directories found: {len(directories)}")

    counts = {"created": 0, "existing": 0, "failed": 0}
    failures: list[str] = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_directory,
                str(directory),
                args.gap,
                args.overwrite,
            ): directory
            for directory in directories
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Creating strips",
            unit="image",
        ):
            directory = futures[future]
            try:
                status, detail = future.result()
            except Exception as exc:
                status = "failed"
                detail = f"{directory}: {type(exc).__name__}: {exc}"

            counts[status] += 1
            if status == "failed":
                failures.append(detail)

    print()
    print(f"Created: {counts['created']}")
    print(f"Already existing: {counts['existing']}")
    print(f"Failed: {counts['failed']}")

    if failures:
        failure_path = root / "progressive_strip_failures.txt"
        failure_path.write_text("\n".join(failures), encoding="utf-8")
        print(f"Failure details: {failure_path}")


if __name__ == "__main__":
    main()