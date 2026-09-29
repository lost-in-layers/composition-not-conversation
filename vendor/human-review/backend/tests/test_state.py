import json
import os

from hr_backend.paths import RepoPaths
from hr_backend.state import StateStore


def test_atomic_save_and_reload(fake_repo: RepoPaths):
    store = StateStore(fake_repo, "r1")
    # The slot fixes the reviewer identity.
    assert store.state["reviewer_id"] == "reviewer_1"
    store.set_cursor(3)
    # File on disk is valid JSON and contains the value.
    data = json.loads(fake_repo.review_state_for("r1").read_text(encoding="utf-8"))
    assert data["reviewer_id"] == "reviewer_1"
    assert data["cursor"] == 3
    # No temp files left behind.
    leftovers = [p for p in os.listdir(fake_repo.runtime_dir) if p.endswith(".tmp")]
    assert leftovers == []
    # A fresh store sees the same state (restart simulation).
    assert StateStore(fake_repo, "r1").state["cursor"] == 3
    # The other reviewer slot is a completely separate file.
    assert StateStore(fake_repo, "r2").state["reviewer_id"] == "reviewer_2"
    assert StateStore(fake_repo, "r2").state["cursor"] == 0


def test_draft_roundtrip_and_status(fake_repo: RepoPaths):
    store = StateStore(fake_repo, "r1")
    assert store.scene_status("lvis_x") == "unseen"
    store.save_draft("lvis_x", {"overall_quality": 3})
    assert store.get_draft("lvis_x")["overall_quality"] == 3
    assert store.scene_status("lvis_x") == "draft"
    store.drop_draft("lvis_x")
    assert store.get_draft("lvis_x") is None


def test_defer_and_undefer(fake_repo: RepoPaths):
    store = StateStore(fake_repo, "r1")
    store.defer("lvis_y")
    store.defer("lvis_y")  # idempotent
    assert store.state["deferred"] == ["lvis_y"]
    assert store.scene_status("lvis_y") == "deferred"
    store.undefer("lvis_y")
    assert store.state["deferred"] == []


def test_decision_lifecycle_and_undo(fake_repo: RepoPaths):
    store = StateStore(fake_repo, "r1")
    store.save_draft("s1", {"a": 1})
    store.defer("s1")
    store.record_decision("s1", {"decision": "accepted", "review": {"a": 1}})
    assert store.scene_status("s1") == "accepted"
    assert store.state["drafts"] == {}
    assert store.state["deferred"] == []
    assert store.state["decision_order"] == ["s1"]

    entry = store.undo_last_decision()
    assert entry["scene_id"] == "s1"
    assert store.scene_status("s1") == "draft"
    assert store.get_draft("s1")["restored_from_undo"] is True
    assert store.state["decision_order"] == []
    assert store.undo_last_decision() is None


def test_guide_preference_and_cursor(fake_repo: RepoPaths):
    store = StateStore(fake_repo, "r1")
    store.set_guide_preference(True)
    store.set_cursor(5)
    reloaded = StateStore(fake_repo, "r1")
    assert reloaded.state["guide"]["do_not_autoshow"] is True
    assert reloaded.state["cursor"] == 5


def test_backup_creates_timestamped_copy(fake_repo: RepoPaths):
    store = StateStore(fake_repo, "r1")
    store.set_cursor(2)
    target = store.backup()
    assert target.exists()
    assert target.parent == fake_repo.backup_dir
    assert target.name.startswith("review_state_r1.")
    assert json.loads(target.read_text(encoding="utf-8"))["reviewer_id"] == "reviewer_1"


def test_v2_migration_reopens_accepted_scene_missing_answers(fake_repo: RepoPaths):
    fake_repo.ensure_runtime_dirs()
    path = fake_repo.review_state_for("r1")
    old = {
        "schema_version": 2,
        "reviewer_slot": "r1",
        "reviewer_id": "reviewer_1",
        "guide": {"do_not_autoshow": False},
        "cursor": 7,
        "drafts": {},
        "deferred": [],
        "decisions": {
            "scene_a": {
                "decision": "accepted",
                "review": {
                    "primary_category": "direct_ask",
                    "suitable_categories": ["direct_ask"],
                    "human_questions": {"direct_ask": "What is visible?"},
                },
            },
            "scene_b": {"decision": "rejected", "review": {}},
        },
        "decision_order": ["scene_a", "scene_b"],
        "undo_history": [],
    }
    path.write_text(json.dumps(old), encoding="utf-8")

    store = StateStore(fake_repo, "r1")
    assert store.state["schema_version"] == 3
    assert store.scene_status("scene_a") == "draft"
    assert store.scene_status("scene_b") == "rejected"
    assert store.get_draft("scene_a")["human_questions"] == {"direct_ask": "What is visible?"}
    assert store.get_draft("scene_a")["human_answers"] == {}
    assert store.state["decision_order"] == ["scene_b"]
    assert store.state["cursor"] == 0
    assert list(fake_repo.backup_dir.glob("*.pre-v3.json"))
