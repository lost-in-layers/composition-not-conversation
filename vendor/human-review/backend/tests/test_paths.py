import pytest

from hr_backend.paths import RepoPaths, find_repo_root


def test_resolve_servable_ok(fake_repo: RepoPaths):
    p = fake_repo.resolve_servable("data/pre-data/renders/lvis/lvis/000000000001/full.png")
    assert p.exists()
    assert p.suffix == ".png"


def test_resolve_servable_backslashes(fake_repo: RepoPaths):
    p = fake_repo.resolve_servable("data\\pre-data\\renders\\lvis\\lvis\\000000000001\\full.png")
    assert p.exists()


@pytest.mark.parametrize(
    "bad",
    [
        "../outside.png",
        "data/../../etc/passwd.png",
        "/etc/passwd.png",
        "C:/Windows/system32/config.png",
        "data/derived/mulan-scenes/lvis_scenes.json",  # not an image
        "src/somefile.png",  # outside data/
        "",
    ],
)
def test_resolve_servable_rejects(fake_repo: RepoPaths, bad: str):
    with pytest.raises(ValueError):
        fake_repo.resolve_servable(bad)


def test_find_repo_root_env(fake_repo: RepoPaths, monkeypatch):
    monkeypatch.setenv("HR_REPO_ROOT", str(fake_repo.repo_root))
    assert find_repo_root() == fake_repo.repo_root


def test_repo_relative_roundtrip(fake_repo: RepoPaths):
    p = fake_repo.repo_root / "data" / "x" / "y.png"
    assert fake_repo.to_repo_relative(p) == "data/x/y.png"
