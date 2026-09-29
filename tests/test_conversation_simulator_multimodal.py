from __future__ import annotations

import base64
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace


VENDOR_DIR = Path(__file__).resolve().parents[1] / "vendor" / "conv-smi"
if str(VENDOR_DIR) not in sys.path:
    sys.path.append(str(VENDOR_DIR))

from multimodal import (  # noqa: E402
    content_to_text,
    image_display_source,
    materialize_local_images,
    multimodal_content,
    sample_images,
)


def test_text_only_content_is_backwards_compatible():
    assert multimodal_content("hello") == "hello"


def test_local_image_is_encoded_relative_to_dataset_directory(tmp_path):
    image_bytes = b"not-a-real-png-but-valid-for-transport"
    (tmp_path / "scene.png").write_bytes(image_bytes)
    messages = [
        {
            "role": "user",
            "content": multimodal_content("What is shown?", ["scene.png"]),
        }
    ]

    prepared = materialize_local_images(messages, image_base_dir=tmp_path)
    image_url = prepared[0]["content"][1]["image_url"]["url"]

    assert image_url == (
        "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    )
    assert messages[0]["content"][1]["image_url"]["url"] == "scene.png"


def test_remote_url_and_detail_are_preserved():
    messages = [
        {
            "role": "user",
            "content": multimodal_content(
                "Compare these",
                [{"url": "https://example.test/image.webp", "detail": "high"}],
            ),
        }
    ]

    prepared = materialize_local_images(messages)

    assert prepared[0]["content"][1] == {
        "type": "image_url",
        "image_url": {
            "url": "https://example.test/image.webp",
            "detail": "high",
        },
    }


def test_display_source_resolves_dataset_relative_path(tmp_path):
    expected = tmp_path / "images" / "scene.png"
    assert image_display_source("images/scene.png", tmp_path) == str(expected)


def test_sample_and_shard_images_are_selected_per_turn():
    sample = {
        "images": ["shared.jpg"],
        "shards": [
            {"shard_id": 1, "shard": "first", "images": ["crop-1.png"]},
            {"shard_id": 2, "shard": "second", "images": ["crop-2.png"]},
        ],
    }

    assert sample_images(sample, 1) == ["shared.jpg", "crop-1.png"]
    assert sample_images(sample, 2, include_sample=False) == ["crop-2.png"]


def test_transcript_renders_images_without_dumping_base64():
    content = multimodal_content(
        "question",
        ["data:image/png;base64,abcdefghijklmnopqrstuvwxyz"],
    )

    assert content_to_text(content) == "question\n[image: data:image/png;base64,...]"


class _FakeResponse:
    def to_dict(self):
        return {
            "choices": [{"message": {"content": "answer"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            },
        }


class _FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _FakeResponse()


def test_generate_sends_openai_compatible_multimodal_payload(tmp_path, monkeypatch):
    (tmp_path / "scene.jpg").write_bytes(b"jpeg-ish")
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    sys.modules.pop("model_openai", None)
    model_openai = importlib.import_module("model_openai")
    model = model_openai.OpenAI_Model()
    completions = _FakeCompletions()
    # Setting `client` bypasses backend routing and sends everything here.
    model.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    result = model.generate(
        [{"role": "user", "content": "What is shown?"}],
        model="local-vlm",
        images=["scene.jpg"],
        image_base_dir=tmp_path,
        return_metadata=True,
    )

    sent_content = completions.kwargs["messages"][0]["content"]
    assert sent_content[0] == {"type": "text", "text": "What is shown?"}
    assert sent_content[1]["type"] == "image_url"
    assert sent_content[1]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )
    assert result["message"] == "answer"
