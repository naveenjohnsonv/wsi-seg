from pathlib import Path

import pytest

from wsi_seg.config import AppConfig

BASE = {
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


def test_paths_defaults_when_omitted() -> None:
    cfg = AppConfig.model_validate(BASE)
    assert cfg.paths.slide_path is None
    assert cfg.paths.model_path == Path("data/model_scripted.pt")
    assert cfg.paths.output_dir == Path("outputs")
    assert cfg.output.keep_memmap is False


def test_paths_from_yaml_when_provided() -> None:
    with_paths = dict(BASE)
    with_paths["paths"] = {
        "slide_path": "data/slides/test.mrxs",
        "model_path": "data/custom_model.pt",
        "output_dir": "my_outputs",
    }
    cfg = AppConfig.model_validate(with_paths)
    assert cfg.paths.slide_path == Path("data/slides/test.mrxs")
    assert cfg.paths.model_path == Path("data/custom_model.pt")
    assert cfg.paths.output_dir == Path("my_outputs")


def test_from_yaml_resolves_paths_relative_to_config_dir(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    cfg_path = cfg_dir / "cfg.yaml"
    cfg_path.write_text(
        """
paths:
  slide_path: ../data/slide.mrxs
  model_path: ../data/model.pt
  output_dir: ../outputs/run
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
    assert cfg.paths.slide_path == (tmp_path / "data" / "slide.mrxs").resolve()
    assert cfg.paths.model_path == (tmp_path / "data" / "model.pt").resolve()
    assert cfg.paths.output_dir == (tmp_path / "outputs" / "run").resolve()


def test_from_yaml_absolute_paths_unchanged(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
paths:
  slide_path: /absolute/slide.mrxs
  model_path: /absolute/model.pt
  output_dir: /absolute/outputs
model:
  patch_px: 512
  stride_px: 384
  halo_px: 64
""",
        encoding="utf-8",
    )
    cfg = AppConfig.from_yaml(cfg_path)
    assert cfg.paths.slide_path == Path("/absolute/slide.mrxs")
    assert cfg.paths.model_path == Path("/absolute/model.pt")
    assert cfg.paths.output_dir == Path("/absolute/outputs")


def test_from_yaml_no_paths_section_defaults_resolve_to_cwd(tmp_path: Path) -> None:
    subdir = tmp_path / "nested" / "configs"
    subdir.mkdir(parents=True)
    cfg_path = subdir / "minimal.yaml"
    cfg_path.write_text(
        """
model:
  patch_px: 512
  stride_px: 384
  halo_px: 64
""",
        encoding="utf-8",
    )
    cfg = AppConfig.from_yaml(cfg_path)
    cwd = Path.cwd()
    assert cfg.paths.slide_path is None
    assert cfg.paths.model_path == (cwd / "data" / "model_scripted.pt").resolve()
    assert cfg.paths.output_dir == (cwd / "outputs").resolve()


def test_from_yaml_upgrades_legacy_future_schedule(tmp_path: Path) -> None:
    cfg_path = tmp_path / "legacy.yaml"
    cfg_path.write_text(
        """
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
