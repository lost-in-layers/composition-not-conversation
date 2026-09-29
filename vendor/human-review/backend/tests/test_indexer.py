import json
import os
import time

import pytest

from hr_backend import indexer
from hr_backend.atomicio import read_json
from hr_backend.paths import RepoPaths


def test_build_cache_offsets_and_fetch(fake_repo: RepoPaths):
    index = indexer.build_cache(fake_repo)
    assert index["subsets"]["lvis"]["count"] == 4
    assert index["subsets"]["laion"]["count"] == 3

    # Every recorded offset/length must slice exactly one scene line.
    for scene_id, entry in index["scenes"].items():
        cache_path = fake_repo.cache_dir / entry["file"]
        with open(cache_path, "rb") as handle:
            handle.seek(entry["offset"])
            raw = handle.read(entry["length"])
        scene = json.loads(raw)
        assert scene["scene_id"] == scene_id

    scene = indexer.fetch_scene(fake_repo, index, "lvis_000000000002")
    assert scene["scene_id"] == "lvis_000000000002"
    assert len(scene["layers"]) == 3
    # Technical nulls survive the cache untouched.
    assert scene["layers"][0]["label"] is None
    assert scene["layers"][0]["mask"]["matches_native_alpha"] is None


def test_fetch_unknown_scene(fake_repo: RepoPaths):
    index = indexer.build_cache(fake_repo)
    with pytest.raises(KeyError):
        indexer.fetch_scene(fake_repo, index, "lvis_does_not_exist")


def test_header_metadata(fake_repo: RepoPaths):
    index = indexer.build_cache(fake_repo)
    header = index["subsets"]["lvis"]["header"]
    assert header.get("subset") == "lvis"
    assert "geometry_policy" in header


def test_freshness_warnings(fake_repo: RepoPaths):
    index = indexer.build_cache(fake_repo)
    assert indexer.source_freshness_warnings(fake_repo, index) == []

    src = fake_repo.sources["lvis"]
    text = src.read_text(encoding="utf-8")
    time.sleep(0.01)
    src.write_text(text + " ", encoding="utf-8")
    warnings = indexer.source_freshness_warnings(fake_repo, index)
    assert any("lvis" in w for w in warnings)


def test_rebuild_preserves_review_state(fake_repo: RepoPaths):
    fake_repo.ensure_runtime_dirs()
    state_path = fake_repo.review_state_for("r1")
    state_path.write_text('{"schema_version": 2, "decisions": {"x": 1}}', encoding="utf-8")
    indexer.build_cache(fake_repo)
    indexer.build_cache(fake_repo)  # rebuild again
    assert read_json(state_path)["decisions"] == {"x": 1}


def test_no_leftover_temp_files(fake_repo: RepoPaths):
    indexer.build_cache(fake_repo)
    leftovers = [p for p in os.listdir(fake_repo.cache_dir) if p.endswith(".tmp")]
    assert leftovers == []


def test_excluded_scenes_not_indexed(fake_repo: RepoPaths):
    """Source excluded_scenes arrays (no renders) must not enter the queue."""
    index = indexer.build_cache(fake_repo)
    assert "lvis_excluded" not in index["scenes"]
    assert "laion_excluded" not in index["scenes"]


def test_only_selected_scenes_are_indexed(fake_repo: RepoPaths):
    """A scene in the source JSON but absent from data/selected_1200 is skipped."""
    import shutil

    shutil.rmtree(fake_repo.selected_dir / "lvis" / "000000000003")
    index = indexer.build_cache(fake_repo)
    assert "lvis_000000000003" not in index["scenes"]
    assert index["subsets"]["lvis"]["count"] == 3
    assert index["selection"]["found"]["lvis"] == 3


def test_images_repointed_to_selected_1200(fake_repo: RepoPaths):
    index = indexer.build_cache(fake_repo)
    scene = indexer.fetch_scene(fake_repo, index, "lvis_000000000001")
    assert scene["render_dir"] == "data/selected_1200/lvis/000000000001"
    assert scene["full_path"] == "data/selected_1200/lvis/000000000001/full.png"
    assert scene["background"]["path"] == "data/selected_1200/lvis/000000000001/layer_0.png"
    for layer in scene["layers"]:
        assert layer["path"].startswith("data/selected_1200/lvis/000000000001/")
        assert (fake_repo.repo_root / layer["path"]).exists()


def test_selection_warns_on_missing_metadata(fake_repo: RepoPaths):
    """A selected scene with no record in the source JSON is reported."""
    extra = fake_repo.selected_dir / "lvis" / "000000000999"
    extra.mkdir(parents=True)
    (extra / "full.png").write_bytes(b"")
    index = indexer.build_cache(fake_repo)
    assert "lvis_000000000999" in index["selection"]["missing_ids"]["lvis"]
    assert any("000000000999" in w for w in indexer.selection_warnings(index))


def test_no_selection_raises(fake_repo: RepoPaths):
    import shutil

    shutil.rmtree(fake_repo.selected_dir)
    with pytest.raises(FileNotFoundError):
        indexer.build_cache(fake_repo)
