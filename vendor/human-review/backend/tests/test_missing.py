from conftest import make_scene

from hr_backend.missing import classify_scene


def fields(bucket):
    return {(item["field"], item.get("layer_id")) for item in bucket}


def test_null_labels_are_human_required():
    scene = make_scene("lvis", "000000000001", n_layers=2)
    report = classify_scene(scene)
    hr = fields(report["human_required"])
    assert ("label", 1) in hr and ("label", 2) in hr
    assert ("label_source", 1) in hr
    assert ("reviewed_image_caption", None) in hr
    assert ("reviewed_image_caption_llava", None) in hr
    assert ("reviewed_background_caption", None) in hr
    assert ("reviewed_background_caption_llava", None) in hr
    assert ("human_questions", None) in hr
    assert ("primary_category", None) in hr


def test_technical_nulls_are_not_human_required():
    scene = make_scene("lvis", "000000000001", n_layers=1)
    report = classify_scene(scene)
    tech = fields(report["technical_ok"])
    assert ("annotation.actual_sha256", None) in tech
    assert ("annotation.matches_inventory_sha256", None) in tech
    assert ("mask.native_alpha_sha256", 1) in tech
    assert ("mask.matches_native_alpha", 1) in tech
    assert ("inventory.error", None) in tech
    # None of those may appear in the human-required bucket.
    hr_fields = {item["field"] for item in report["human_required"]}
    assert not any(f.startswith(("annotation.", "mask.", "inventory.")) for f in hr_fields)


def test_empty_categories_technical_for_laion():
    scene = make_scene("laion", "000000101", n_layers=1)
    report = classify_scene(scene)
    assert ("categories", None) in fields(report["technical_ok"])


def test_layer_without_source_caption_is_blocking_until_excluded_or_written():
    scene = make_scene("lvis", "000000000004", n_layers=2, layer_without_caption=2)
    report = classify_scene(scene)
    assert ("captions", 2) in fields(report["blocking"])

    # Reviewer writes a caption -> no longer blocking.
    review = {"layer_reviews": {"2": {"decision": "keep", "caption": "a written caption"}}}
    report = classify_scene(scene, review)
    assert ("captions", 2) in fields(report["blocking"])  # source itself is still empty
    # ...but the acceptance path treats it as resolved via the review overlay;
    # classification keeps reporting the SOURCE gap for transparency.

    # Excluding the layer moves it to technical_ok (moot).
    review = {"layer_reviews": {"2": {"decision": "exclude", "exclusion_reason": "meaningless_layer"}}}
    report = classify_scene(scene, review)
    assert ("captions", 2) in fields(report["technical_ok"])
    assert ("captions", 2) not in fields(report["blocking"])


def test_draft_overlay_clears_human_required():
    scene = make_scene("lvis", "000000000001", n_layers=1)
    review = {
        "layer_reviews": {"1": {"decision": "keep", "label": "giraffe", "caption": "a giraffe", "geometry_status": "correct"}},
        "reviewed_image_caption": "full caption",
        "reviewed_image_caption_llava": "full LLaVA caption",
        "reviewed_background_caption": "background caption",
        "reviewed_background_caption_llava": "background LLaVA caption",
        "primary_category": "direct_ask",
        "suitable_categories": ["direct_ask"],
        "category_decision_made": True,
        "human_questions": {"direct_ask": "What is shown?"},
        "human_answers": {"direct_ask": "An object."},
    }
    report = classify_scene(scene, review)
    assert report["counts"]["human_required"] == 0
    assert report["counts"]["blocking"] == 0


def test_excluded_layer_needs_no_label():
    scene = make_scene("lvis", "000000000001", n_layers=2)
    review = {"layer_reviews": {"2": {"decision": "exclude", "exclusion_reason": "duplicate"}}}
    report = classify_scene(scene, review)
    assert ("label", 2) not in fields(report["human_required"])
    assert ("label", 2) in fields(report["technical_ok"])
    # Layer 1 is still pending.
    assert ("label", 1) in fields(report["human_required"])


def test_per_layer_counts():
    scene = make_scene("lvis", "000000000001", n_layers=2)
    report = classify_scene(scene)
    assert report["per_layer"]["1"]["human_required"] > 0
    assert report["per_layer"]["1"]["technical_ok"] > 0


def test_missing_file_detection(fake_repo):
    scene = make_scene("lvis", "000000000001", n_layers=2)
    (fake_repo.repo_root / scene["layers"][0]["path"]).unlink()
    report = classify_scene(scene, paths=fake_repo, check_files=True)
    assert ("path", 1) in fields(report["blocking"])
