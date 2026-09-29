"""Utilities for text-and-image chat messages.

Dataset records keep image references readable (paths or URLs). Immediately
before an API call, local paths are converted to data URLs so a vLLM server
does not need access to the client's filesystem.
"""

from __future__ import annotations

import base64
import copy
import mimetypes
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


IMAGE_DETAIL_VALUES = {"auto", "low", "high"}


def _as_image_list(images: Any) -> list[Any]:
    if images is None:
        return []
    if isinstance(images, (str, Path, dict)):
        return [images]
    if isinstance(images, (list, tuple)):
        return list(images)
    raise TypeError("images must be an image reference or a list of references")


def image_content_part(image: Any) -> dict[str, Any]:
    """Convert a friendly image reference to an OpenAI-compatible part.

    Accepted references are a path/URL string, ``{"path": ...}``,
    ``{"url": ...}``, or an already formed ``image_url`` content part.
    """
    detail = "auto"
    if isinstance(image, Path):
        image = str(image)

    if isinstance(image, str):
        url = image
    elif isinstance(image, dict):
        if image.get("type") == "image_url":
            image_url = image.get("image_url")
            if isinstance(image_url, str):
                url = image_url
            elif isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                url = image_url["url"]
                detail = image_url.get("detail", image.get("detail", "auto"))
            else:
                raise ValueError("image_url content parts require a URL string")
        else:
            url = image.get("url", image.get("path"))
            detail = image.get("detail", "auto")
            if not isinstance(url, str):
                raise ValueError("image references require a string 'path' or 'url'")
    else:
        raise TypeError("each image reference must be a path, URL, or dictionary")

    if not url.strip():
        raise ValueError("image references cannot be empty")
    if detail not in IMAGE_DETAIL_VALUES:
        raise ValueError("image detail must be one of: auto, low, high")
    return {
        "type": "image_url",
        "image_url": {"url": url, "detail": detail},
    }


def multimodal_content(text: Any, images: Any = None) -> Any:
    """Return string content for text-only input or content parts for images."""
    image_parts = [image_content_part(image) for image in _as_image_list(images)]
    if not image_parts:
        return text

    if isinstance(text, str):
        parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    elif isinstance(text, list):
        parts = copy.deepcopy(text)
    else:
        raise TypeError("multimodal message text must be a string or content-part list")
    return parts + image_parts


def sample_images(sample: dict[str, Any], shard_id: Any = None, include_sample: bool = True) -> list[Any]:
    """Collect top-level and optional per-shard image references from a sample."""
    images: list[Any] = []
    if include_sample:
        images.extend(_as_image_list(sample.get("images")))
    if shard_id is not None:
        for shard in sample.get("shards", []):
            if shard.get("shard_id") == shard_id:
                images.extend(_as_image_list(shard.get("images")))
                break
    return images


def describe_images(images: Any) -> str:
    """Name the images a turn is about to attach, using their `label` keys.

    Content parts carry no filename, so a model cannot tell which decomposed
    layer it is looking at unless the accompanying text says so. Datasets that
    do not set labels get an empty string and unchanged turn text.
    """
    labels = [
        image["label"]
        for image in _as_image_list(images)
        if isinstance(image, dict) and image.get("label")
    ]
    if not labels:
        return ""
    if len(labels) == 1:
        return f"(sharing {labels[0]})"
    return "(sharing " + ", ".join(labels[:-1]) + f" and {labels[-1]})"


def all_sample_images(sample: dict[str, Any]) -> list[Any]:
    images = sample_images(sample)
    for shard in sample.get("shards", []):
        images.extend(_as_image_list(shard.get("images")))
    return images


def _local_path_from_url(url: str, base_dir: Path) -> Path | None:
    if re.match(r"^[A-Za-z]:[\\/]", url):
        return Path(url).expanduser()
    parsed = urlparse(url)
    if parsed.scheme in {"http", "https", "data"}:
        return None
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).expanduser()
    if parsed.scheme:
        raise ValueError(f"unsupported image URL scheme: {parsed.scheme}")
    path = Path(url).expanduser()
    return path if path.is_absolute() else base_dir / path


def image_display_source(image: Any, base_dir: Any = None) -> str:
    """Resolve a friendly image reference for display without base64 encoding."""
    url = image_content_part(image)["image_url"]["url"]
    local_path = _local_path_from_url(url, Path(base_dir or ".").resolve())
    return str(local_path.resolve()) if local_path is not None else url


def _path_to_data_url(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"image file not found: {resolved}")
    media_type, _ = mimetypes.guess_type(resolved.name)
    if not media_type or not media_type.startswith("image/"):
        raise ValueError(f"could not determine image MIME type: {resolved}")
    encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def materialize_local_images(messages: list[dict[str, Any]], image_base_dir: Any = None) -> list[dict[str, Any]]:
    """Copy messages and replace local image references with data URLs."""
    prepared = copy.deepcopy(messages)
    base_dir = Path(image_base_dir or ".").resolve()
    for message in prepared:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for index, part in enumerate(content):
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            normalized = image_content_part(part)
            url = normalized["image_url"]["url"]
            local_path = _local_path_from_url(url, base_dir)
            if local_path is not None:
                normalized["image_url"]["url"] = _path_to_data_url(local_path)
            content[index] = normalized
    return prepared


def content_to_text(content: Any, image_placeholder: str | None = None) -> str:
    """Render content parts for terminal output, logs, and prompts.

    `image_placeholder` replaces the default `[image: <url>]` rendering. Pass
    `"[image]"` for any text that reaches an agent prompt: image filenames
    (`.../layer_2.png`) can name the answer, so paths are only for humans.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    rendered: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            rendered.append(str(part))
        elif part.get("type") == "text":
            rendered.append(str(part.get("text", "")))
        elif part.get("type") == "image_url":
            if image_placeholder is not None:
                rendered.append(image_placeholder)
                continue
            normalized = image_content_part(part)
            url = normalized["image_url"]["url"]
            if url.startswith("data:"):
                url = url.split(",", 1)[0] + ",..."
            rendered.append(f"[image: {url}]")
        else:
            rendered.append(f"[{part.get('type', 'content')}]")
    return "\n".join(value for value in rendered if value)


def image_base_dir(dataset_filename: str | None) -> Path:
    if not dataset_filename:
        return Path.cwd()
    return Path(dataset_filename).expanduser().resolve().parent
