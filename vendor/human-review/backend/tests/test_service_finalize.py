import json

from conftest import make_valid_review

from hr_backend.atomicio import read_json
from hr_backend.curation import validate_curated_scene
from hr_backend.exports import write_exports
from hr_backend.service import ReviewService


def first_scene(service: ReviewService, subset=None):
    sid = service.current_scene_id()
    if subset:
        for candidate in service.queue["order"]:
            if candidate.startswith(subset) and not service.store.decision_for(candidate):
                sid = candidate
                break
    return sid, service.get_scene(sid)["scene"]


def test_accept_full_flow_and_export(service: ReviewService):
    sid, scene = first_scene(service)
    review = make_valid_review(scene)
    result = service.finalize(sid, review, "accepted")
    assert result["ok"], result.get("problems")

    exported = read_json(service.paths.accepted_export_for(service.reviewer))
    assert exported["count"] == 1
    curated = exported["scenes"][0]
    assert curated["scene_id"] == sid
    assert validate_curated_scene(curated) == []

    for layer in curated["layers"]:
        assert layer["label_source"] == "human_review"
        assert layer["label"]
        assert layer["caption"]
        assert layer["caption_quality"] == "human_verified"
    assert curated["captions"]["image"] == "a reviewed full-image caption"
    assert curated["captions"]["background"] == "a reviewed background caption"
    assert curated["captions"]["image_by_model"]["llava"] == "a reviewed LLaVA full-image caption"
    assert curated["captions"]["background_by_model"]["llava"] == "a reviewed LLaVA background caption"
    assert curated["questions"] == [
        {"category": "direct_ask", "question": "What object is visible in the scene?", "answer": "An object is visible."}
    ]
    assert curated["review"]["human_questions"] == {"direct_ask": "What object is visible in the scene?"}
    assert curated["review"]["human_answers"] == {"direct_ask": "An object is visible."}
    # Original nulls preserved in provenance, untouched.
    for original in curated["provenance"]["original_layers"]:
        assert original["label"] is None
        assert original["label_source"] is None
    assert curated["provenance"]["annotation"]["actual_sha256"] is None


def test_export_upgrades_legacy_question_snapshot_from_saved_review(service: ReviewService):
    sid, scene = first_scene(service)
    result = service.finalize(sid, make_valid_review(scene), "accepted")
    assert result["ok"]
    record = service.store.decision_for(sid)
    for question in record["curated_scene"]["questions"]:
        question.pop("answer", None)

    export = write_exports(service.paths, service.store.state, service.reviewer)
    assert export["accepted"] == 1
    assert export["problems"] == []
    accepted = read_json(service.paths.accepted_export_for(service.reviewer))["scenes"]
    assert accepted[0]["questions"][0]["answer"] == "An object is visible."


def test_accept_with_layer_exclusion_filters_relations(service: ReviewService):
    sid = "lvis_000000000002"  # 3 layers with relations + occlusion
    scene = service.get_scene(sid)["scene"]
    review = make_valid_review(scene, exclude=[2])
    result = service.finalize(sid, review, "accepted")
    assert result["ok"], result.get("problems")

    record = service.store.decision_for(sid)
    curated = record["curated_scene"]
    active_ids = {layer["layer_id"] for layer in curated["layers"]}
    assert active_ids == {1, 3}
    assert {e["layer_id"] for e in curated["excluded_layers"]} == {2}
    for rel in curated["relations"]:
        assert rel["subject_layer_id"] in active_ids
        assert rel["object_layer_id"] in active_ids
    for pair in curated["pairwise_geometry"]:
        assert set(pair["layer_ids"]) <= active_ids
    # Excluded layer keeps its provenance entry.
    assert len(curated["provenance"]["original_layers"]) == 3


def test_accept_multiple_categories_with_one_question(service: ReviewService):
    sid, scene = first_scene(service)
    review = make_valid_review(scene)
    review["suitable_categories"] = ["direct_ask", "spatial", "counting"]
    review["human_questions"] = {"spatial": "Which object is above the other object?"}
    review["human_answers"] = {"spatial": "The first object."}
    result = service.finalize(sid, review, "accepted")
    assert result["ok"], result.get("problems")
    curated = service.store.decision_for(sid)["curated_scene"]
    assert curated["questions"] == [
        {"category": "spatial", "question": "Which object is above the other object?", "answer": "The first object."}
    ]


def test_accept_validation_failures(service: ReviewService):
    sid, scene = first_scene(service)

    # Missing label on a retained layer.
    review = make_valid_review(scene)
    review["layer_reviews"]["1"]["label"] = "  "
    result = service.finalize(sid, review, "accepted")
    assert not result["ok"]
    assert any("canonical short label" in p for p in result["problems"])
    assert service.store.decision_for(sid) is None

    # All layers excluded -> unacceptable.
    review = make_valid_review(scene, exclude=[l["layer_id"] for l in scene["layers"]])
    result = service.finalize(sid, review, "accepted")
    assert not result["ok"]
    assert any("At least one retained" in p for p in result["problems"])

    # Primary category not in suitable list.
    review = make_valid_review(scene)
    review["suitable_categories"] = []
    result = service.finalize(sid, review, "accepted")
    assert not result["ok"]
    assert any("primary category must also be marked as suitable" in p for p in result["problems"])

    # Every filled question requires an answer.
    review = make_valid_review(scene)
    review["human_answers"] = {}
    result = service.finalize(sid, review, "accepted")
    assert not result["ok"]
    assert any("answer is required" in p.lower() for p in result["problems"])

    # Automatic evidence hints never block a human category choice.
    review = make_valid_review(scene, primary="multi_hop_reference")
    result = service.finalize(sid, review, "accepted")
    assert result["ok"], result.get("problems")
    service.undo_last()

    # Geometry major problem blocks acceptance.
    review = make_valid_review(scene)
    review["layer_reviews"]["1"]["geometry_status"] = "major_problem"
    result = service.finalize(sid, review, "accepted")
    assert not result["ok"]

    # Invalid bbox override (x1 >= x2).
    review = make_valid_review(scene)
    review["layer_reviews"]["1"]["bbox_override"] = [30, 10, 20, 40]
    result = service.finalize(sid, review, "accepted")
    assert not result["ok"]
    assert any("bbox" in p for p in result["problems"])

    # Out-of-bounds bbox override.
    review = make_valid_review(scene)
    review["layer_reviews"]["1"]["bbox_override"] = [0, 0, 9999, 10]
    result = service.finalize(sid, review, "accepted")
    assert not result["ok"]


def test_geometry_override_stored_separately(service: ReviewService):
    sid, scene = first_scene(service)
    review = make_valid_review(scene)
    review["layer_reviews"]["1"]["geometry_status"] = "minor_correction"
    review["layer_reviews"]["1"]["bbox_override"] = [1, 1, 20, 20]
    review["layer_reviews"]["1"]["centroid_override"] = [10, 10]
    result = service.finalize(sid, review, "accepted")
    assert result["ok"], result.get("problems")
    curated = service.store.decision_for(sid)["curated_scene"]
    layer1 = next(l for l in curated["layers"] if l["layer_id"] == 1)
    # Original geometry untouched; override lives next to it.
    assert layer1["geometry"]["bbox_xyxy"] == scene["layers"][0]["geometry"]["bbox_xyxy"]
    assert layer1["geometry_review"]["bbox_override_xyxy"] == [1, 1, 20, 20]
    assert layer1["geometry_review"]["centroid_override_xy"] == [10, 10]
    assert layer1["geometry_review"]["status"] == "minor_correction"


def test_blocking_source_caption_requires_reviewer_caption(service: ReviewService):
    sid = "lvis_000000000004"  # layer 2 has no source caption
    scene = service.get_scene(sid)["scene"]
    review = make_valid_review(scene)
    result = service.finalize(sid, review, "accepted")
    # Reviewer wrote captions for every retained layer, so this is acceptable.
    assert result["ok"], result.get("problems")


def test_reject_flow(service: ReviewService):
    sid, scene = first_scene(service)
    result = service.finalize(sid, {}, "rejected")
    assert result["ok"]
    exported = read_json(service.paths.rejected_export_for(service.reviewer))
    assert exported["count"] == 1
    rec = exported["scenes"][0]
    assert rec["scene_id"] == sid
    assert rec["reviewer_id"] == "reviewer_1"
    assert "missing_field_findings" in rec
    # Rejected scenes never enter the accepted export.
    assert read_json(service.paths.accepted_export_for(service.reviewer))["count"] == 0


def test_quota_accounting_and_subset_counts(service: ReviewService):
    sid_lvis, scene_lvis = first_scene(service, "lvis")
    service.finalize(sid_lvis, make_valid_review(scene_lvis, primary="direct_ask"), "accepted")
    sid_laion, scene_laion = first_scene(service, "laion")
    service.finalize(sid_laion, make_valid_review(scene_laion, primary="spatial"), "accepted")

    prog = service.get_progress()
    assert prog["accepted"] == 2
    assert prog["category_counts"]["direct_ask"] == 1
    assert prog["category_counts"]["spatial"] == 1
    assert prog["accepted_by_subset"] == {"lvis": 1, "laion": 1}
    assert prog["reviewed"] == 2
    assert prog["remaining_to_target"] == prog["queue_total"] - 2


def test_undo_restores_draft_and_rebuilds_exports(service: ReviewService):
    sid, scene = first_scene(service)
    service.finalize(sid, make_valid_review(scene), "accepted")
    assert read_json(service.paths.accepted_export_for(service.reviewer))["count"] == 1

    result = service.undo_last()
    assert result["ok"] and result["scene_id"] == sid
    assert service.store.decision_for(sid) is None
    assert service.store.get_draft(sid) is not None
    assert read_json(service.paths.accepted_export_for(service.reviewer))["count"] == 0
    # The scene is current again.
    assert service.current_scene_id() == sid


def test_defer_and_resume_after_restart(service: ReviewService):
    first = service.current_scene_id()
    service.defer_scene(first)
    second = service.current_scene_id()
    assert second != first

    scene = service.get_scene(second)["scene"]
    service.finalize(second, make_valid_review(scene), "accepted")
    third = service.current_scene_id()

    # Restart: a brand-new service over the same files resumes identically.
    restarted = ReviewService(service.paths, seed=20260719)
    assert restarted.current_scene_id() == third
    assert restarted.queue["order"] == service.queue["order"]
    assert restarted.store.state["deferred"] == [first]


def test_jump_does_not_move_cursor(service: ReviewService):
    cursor_before = service.store.state["cursor"]
    other = service.queue["order"][2]
    payload = service.get_scene(other)  # jump = plain fetch
    assert payload["scene"]["scene_id"] == other
    assert service.store.state["cursor"] == cursor_before


def test_draft_then_finalize_conflict(service: ReviewService):
    import pytest

    sid, scene = first_scene(service)
    service.save_draft(sid, {"overall_quality": 2})
    assert service.get_scene(sid)["status"] == "draft"
    service.finalize(sid, make_valid_review(scene), "accepted")
    with pytest.raises(ValueError):
        service.save_draft(sid, {"overall_quality": 1})
    with pytest.raises(ValueError):
        service.finalize(sid, make_valid_review(scene), "accepted")
