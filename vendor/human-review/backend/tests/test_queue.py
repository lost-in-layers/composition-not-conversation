import json

from hr_backend import indexer, queueing
from hr_backend.paths import RepoPaths


def test_assignments_disjoint_and_cover_all(fake_repo: RepoPaths):
    index = indexer.build_cache(fake_repo)
    a = queueing.assignments(index, seed=20260719)
    b = queueing.assignments(index, seed=20260719)
    assert a == b  # deterministic

    r1 = set(a["r1"])
    r2 = set(a["r2"])
    assert r1.isdisjoint(r2)  # each reviewer's images are fixed and non-overlapping
    all_ids = {sid for subset in index["subsets"].values() for sid in subset["scene_ids"]}
    assert r1 | r2 == all_ids  # together they cover the whole selection once
    assert len(a["r1"]) + len(a["r2"]) == 7


def test_each_reviewer_interleaves_lvis_first(fake_repo: RepoPaths):
    index = indexer.build_cache(fake_repo)
    a = queueing.assignments(index, seed=20260719)
    # r1 gets 2 LVIS + 2 LAION, interleaved LVIS-first.
    subsets_r1 = ["lvis" if sid.startswith("lvis_") else "laion" for sid in a["r1"]]
    assert subsets_r1 == ["lvis", "laion", "lvis", "laion"]
    # r2 gets 2 LVIS + 1 LAION.
    subsets_r2 = ["lvis" if sid.startswith("lvis_") else "laion" for sid in a["r2"]]
    assert subsets_r2 == ["lvis", "laion", "lvis"]


def test_queue_seed_changes_assignment(fake_repo: RepoPaths):
    index = indexer.build_cache(fake_repo)
    assert queueing.assignments(index, 20260719) != queueing.assignments(index, 1)


def test_queue_persisted_not_regenerated(fake_repo: RepoPaths):
    index = indexer.build_cache(fake_repo)
    queue = queueing.load_or_create_queue(fake_repo, index, "r1")
    # Tamper with the stored order; a reload must keep the stored version.
    queue["order"] = list(reversed(queue["order"]))
    fake_repo.review_queue_for("r1").write_text(json.dumps(queue), encoding="utf-8")
    reloaded = queueing.load_or_create_queue(fake_repo, index, "r1")
    assert reloaded["order"] == queue["order"]


def test_separate_queue_files_per_reviewer(fake_repo: RepoPaths):
    index = indexer.build_cache(fake_repo)
    q1 = queueing.load_or_create_queue(fake_repo, index, "r1")
    q2 = queueing.load_or_create_queue(fake_repo, index, "r2")
    assert fake_repo.review_queue_for("r1").exists()
    assert fake_repo.review_queue_for("r2").exists()
    assert set(q1["order"]).isdisjoint(set(q2["order"]))


def test_next_unfinished_skips_finalized_and_deferred(fake_repo: RepoPaths):
    index = indexer.build_cache(fake_repo)
    queue = queueing.load_or_create_queue(fake_repo, index, "r1")
    order = queue["order"]
    state = {"decisions": {order[0]: {"decision": "accepted"}}, "deferred": [order[1]]}
    assert queueing.next_unfinished(queue, state, 0) == order[2]

    # When only deferred scenes remain they become the fallback.
    state = {"decisions": {sid: {"decision": "accepted"} for sid in order[1:]}, "deferred": [order[0]]}
    assert queueing.next_unfinished(queue, state, 0) == order[0]

    # Everything finalized -> nothing to show.
    state = {"decisions": {sid: {"decision": "rejected"} for sid in order}, "deferred": []}
    assert queueing.next_unfinished(queue, state, 0) is None


def test_backwards_navigation(fake_repo: RepoPaths):
    index = indexer.build_cache(fake_repo)
    queue = queueing.load_or_create_queue(fake_repo, index, "r1")
    order = queue["order"]
    state = {"decisions": {}, "deferred": []}
    assert queueing.next_unfinished(queue, state, len(order) - 2, direction=-1) == order[len(order) - 2]
    assert queueing.next_unfinished(queue, state, -1, direction=-1) is None
