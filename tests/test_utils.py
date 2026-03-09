import re

from wsi_seg.utils import config_hash, generate_run_id, git_info, utc_now_iso


def test_utc_now_iso_format() -> None:
    ts = utc_now_iso()
    assert re.match(r"^\d{8}T\d{6}Z$", ts), f"unexpected format: {ts}"


def test_git_info_returns_dict_with_expected_keys() -> None:
    info = git_info()
    assert "git_commit" in info
    assert "git_branch" in info
    assert "git_dirty" in info


def test_git_info_returns_sha_in_repo() -> None:
    info = git_info()
    sha = info["git_commit"]
    if sha is not None:
        assert re.match(r"^[0-9a-f]{7}$", sha), f"unexpected sha: {sha}"


def test_config_hash_deterministic() -> None:
    cfg = {
        "model": {"target_mpp": 0.88, "patch_px": 512},
        "schedule": {"use_bounds": True},
    }
    h1 = config_hash(cfg)
    h2 = config_hash(cfg)
    assert h1 == h2
    assert len(h1) == 8
    assert re.match(r"^[0-9a-f]{8}$", h1)


def test_config_hash_changes_with_config() -> None:
    cfg_a = {
        "model": {"target_mpp": 0.88, "patch_px": 512},
        "schedule": {"use_bounds": True},
    }
    cfg_b = {
        "model": {"target_mpp": 0.44, "patch_px": 512},
        "schedule": {"use_bounds": True},
    }
    assert config_hash(cfg_a) != config_hash(cfg_b)


def test_generate_run_id_format() -> None:
    cfg = {
        "model": {"target_mpp": 0.88},
        "schedule": {"use_bounds": True},
    }
    rid = generate_run_id(cfg)
    parts = rid.split("_")
    assert len(parts) == 3, f"expected 3 parts: {rid}"
    assert re.match(r"^\d{8}T\d{6}Z$", parts[0])
    assert len(parts[2]) == 8
