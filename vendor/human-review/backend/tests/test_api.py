from conftest import make_valid_review
from fastapi.testclient import TestClient

from hr_backend.api import create_app
from hr_backend.service import ReviewService


def make_client(service: ReviewService) -> TestClient:
    return TestClient(create_app(service))


def test_bootstrap_and_scene_roundtrip(service: ReviewService):
    client = make_client(service)
    boot = client.get("/api/bootstrap").json()
    assert boot["ready"] is True
    assert boot["reviewer"] == "r1"
    assert boot["reviewer_id"] == "reviewer_1"
    assert boot["seed"] == 20260719

    sid = boot["current_scene_id"]
    payload = client.get(f"/api/scene/{sid}").json()
    assert payload["scene"]["scene_id"] == sid
    assert payload["missing_fields"]["counts"]["human_required"] > 0
    assert len(payload["category_suggestions"]) == 6
    assert payload["file_checks"]["full"] is True


def test_draft_defer_finalize_undo_via_api(service: ReviewService):
    client = make_client(service)
    sid = client.get("/api/queue/current").json()["scene_id"]

    r = client.post(f"/api/scene/{sid}/draft", json={"draft": {"overall_quality": 5}})
    assert r.status_code == 200 and r.json()["saved"]
    # Draft saves return refreshed suggestions so step 4 eligibility tracks
    # the layer decisions made in step 1.
    assert len(r.json()["category_suggestions"]) == 6

    scene1 = client.get(f"/api/scene/{sid}").json()["scene"]
    keep_all = {
        "layer_reviews": {
            str(l["layer_id"]): {"decision": "keep", "label": "x", "caption": "y", "geometry_status": "correct"}
            for l in scene1["layers"]
        }
    }
    r = client.post(f"/api/scene/{sid}/draft", json={"draft": keep_all})
    direct = next(s for s in r.json()["category_suggestions"] if s["category"] == "direct_ask")
    assert direct["eligible"] is True

    r = client.post(f"/api/scene/{sid}/defer")
    assert r.status_code == 200
    next_id = r.json()["next_scene_id"]
    assert next_id != sid

    scene = client.get(f"/api/scene/{next_id}").json()["scene"]
    review = make_valid_review(scene)
    r = client.post(f"/api/scene/{next_id}/finalize", json={"decision": "accepted", "review": review})
    assert r.status_code == 200 and r.json()["ok"]

    # Validation failure -> 422 with problems, no decision recorded.
    third = client.get("/api/queue/current").json()["scene_id"]
    scene3 = client.get(f"/api/scene/{third}").json()["scene"]
    bad = make_valid_review(scene3)
    bad["reviewed_image_caption"] = ""
    r = client.post(f"/api/scene/{third}/finalize", json={"decision": "accepted", "review": bad})
    assert r.status_code == 422
    assert r.json()["problems"]

    r = client.post("/api/undo")
    assert r.status_code == 200 and r.json()["ok"]

    prog = client.get("/api/progress").json()
    assert prog["accepted"] == 0
    dash = client.get("/api/dashboard").json()
    assert "dashboard" in dash


def test_image_serving_and_traversal_protection(service: ReviewService):
    client = make_client(service)
    sid = client.get("/api/queue/current").json()["scene_id"]
    scene = client.get(f"/api/scene/{sid}").json()["scene"]

    ok = client.get(f"/api/image/{scene['full_path']}")
    assert ok.status_code == 200
    assert ok.headers["content-type"] == "image/png"
    assert "max-age" in ok.headers.get("cache-control", "")

    # URL-encoded traversal reaches the route un-normalized and must be blocked.
    assert client.get("/api/image/data/%2e%2e/%2e%2e/etc/passwd.png").status_code == 403
    assert client.get("/api/image/src/secrets.png").status_code == 403
    assert client.get("/api/image/data/derived/mulan-scenes/lvis_scenes.json").status_code == 403
    assert client.get("/api/image/data/pre-data/renders/lvis/lvis/nope/full.png").status_code == 404


def test_reviewer_selection_switch_and_guide_preference(service: ReviewService):
    client = make_client(service)
    # Alias forms are accepted and mapped to a fixed slot.
    r = client.post("/api/reviewer", json={"reviewer": "reviewer_2"})
    assert r.json() == {"reviewer": "r2", "reviewer_id": "reviewer_2"}
    assert client.get("/api/bootstrap").json()["reviewer"] == "r2"

    # Guide preference is stored on the now-active reviewer.
    r = client.post("/api/guide-preference", json={"do_not_autoshow": True})
    assert r.json()["guide"]["do_not_autoshow"] is True
    assert client.get("/api/bootstrap").json()["guide"]["do_not_autoshow"] is True

    # Switching clears the active reviewer, re-gating the app on selection.
    r = client.post("/api/reviewer/switch")
    assert r.json() == {"reviewer": None}
    boot = client.get("/api/bootstrap").json()
    assert boot["ready"] is False and boot["index_ready"] is True
    assert boot["reviewer"] is None
    assert {rv["reviewer"] for rv in boot["reviewers"]} == {"r1", "r2"}

    # An unknown reviewer is a clean 400.
    assert client.post("/api/reviewer", json={"reviewer": "nope"}).status_code == 400


def test_backup_and_export_endpoints(service: ReviewService):
    client = make_client(service)
    r = client.post("/api/backup")
    assert r.status_code == 200 and r.json()["backup_path"].startswith("review_state_r1.")
    r = client.post("/api/export")
    assert r.status_code == 200
    assert service.paths.accepted_export_for("r1").exists()
    assert service.paths.rejected_export_for("r1").exists()


def test_unready_service_returns_409(fake_repo):
    svc = ReviewService(fake_repo)  # no rebuild_index()
    client = make_client(svc)
    assert svc.index is None
    r = client.get("/api/queue/current")
    assert r.status_code == 409
    boot = client.get("/api/bootstrap").json()
    assert boot["ready"] is False and boot["index_ready"] is False
    assert boot["sources"]["lvis"]["exists"] is True
    assert boot["sources"]["selected_1200"]["exists"] is True

    r = client.post("/api/index/rebuild")
    assert r.status_code == 200
    assert r.json()["scenes"] == 7
    # Index is built, but a reviewer must still be selected before it is ready.
    boot = client.get("/api/bootstrap").json()
    assert boot["index_ready"] is True and boot["ready"] is False
    client.post("/api/reviewer", json={"reviewer": "r1"})
    assert client.get("/api/bootstrap").json()["ready"] is True
