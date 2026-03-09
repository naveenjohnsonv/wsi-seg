from pathlib import Path

import pytest

from wsi_seg.config import AppConfig

BASE = {
    "paths": {
        "slide_path": "data/slide.mrxs",
        "model_path": "data/model.pt",
        "output_dir": "outputs/run",
    },
    "model": {
        "target_mpp": 0.88,
        "patch_px": 512,
        "stride_px": 384,
        "halo_px": 64,
        "batch_size": 4,
        "threshold": 0.5,
    },
}


def test_config_validation_accepts_valid_geometry() -> None:
    cfg = AppConfig.model_validate(BASE)
    assert cfg.model.patch_px == 512
    assert cfg.model.stride_px == 384


def test_config_validation_rejects_bad_stride() -> None:
    bad = dict(BASE)
    bad["model"] = dict(BASE["model"])
    bad["model"]["stride_px"] = 400
    with pytest.raises(ValueError):
        AppConfig.model_validate(bad)


def test_from_yaml_resolves_paths(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
paths:
  slide_path: ./data/slide.mrxs
  model_path: ./data/model.pt
  output_dir: ./outputs/run
model:
  patch_px: 512
  stride_px: 384
  halo_px: 64
""",
        encoding="utf-8",
    )
    cfg = AppConfig.from_yaml(cfg_path)
    assert cfg.paths.slide_path.is_absolute()
    assert cfg.paths.model_path.is_absolute()
    assert cfg.paths.output_dir.is_absolute()


def test_from_yaml_upgrades_legacy_future_schedule(tmp_path: Path) -> None:
    cfg_path = tmp_path / "legacy.yaml"
    cfg_path.write_text(
        """
paths:
  slide_path: ./data/slide.mrxs
  model_path: ./data/model.pt
future:
  use_bounds: false
  use_tissue_mask: false
  supertile_px: 8192
""",
        encoding="utf-8",
    )
    cfg = AppConfig.from_yaml(cfg_path)
    assert cfg.schedule.use_bounds is False
    assert cfg.schedule.use_tissue_mask is False
    assert cfg.schedule.supertile_px == 8192
