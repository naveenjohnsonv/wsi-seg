from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    slide_path: Path
    model_path: Path
    output_dir: Path = Path("outputs/run")


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    target_mpp: float = 0.88
    patch_px: int = 512
    stride_px: int = 384
    halo_px: int = 64
    batch_size: int = 8
    threshold: float = 0.5
    apply_sigmoid: bool = True

    @model_validator(mode="after")
    def validate_geometry(self) -> ModelConfig:
        if self.target_mpp <= 0:
            raise ValueError("model.target_mpp must be > 0")
        if self.patch_px <= 0:
            raise ValueError("model.patch_px must be > 0")
        if self.stride_px <= 0:
            raise ValueError("model.stride_px must be > 0")
        if self.halo_px < 0:
            raise ValueError("model.halo_px must be >= 0")
        if self.batch_size <= 0:
            raise ValueError("model.batch_size must be > 0")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("model.threshold must be in [0, 1]")
        valid_extent = self.patch_px - (2 * self.halo_px)
        if valid_extent <= 0:
            raise ValueError("model.patch_px must be greater than 2 * model.halo_px")
        if self.stride_px != valid_extent:
            raise ValueError(
                "For the first implementation, stride_px must equal patch_px - 2 * halo_px "
                "so center-crop stitching covers the output canvas without gaps."
            )
        return self


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    device: str = "cuda"
    use_amp: bool = True
    openslide_cache_bytes: int = 512 * 1024 * 1024
    torch_num_threads: int = 1


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    write_tiff: bool = True
    bigtiff: bool = True
    compression: str | None = None
    preview_max_size: int = 1536
    keep_memmap: bool = True


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    paths: PathsConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    future: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        cfg_path = Path(path)
        with cfg_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        config = cls.model_validate(raw)
        config.paths.slide_path = config.paths.slide_path.expanduser().resolve()
        config.paths.model_path = config.paths.model_path.expanduser().resolve()
        config.paths.output_dir = config.paths.output_dir.expanduser().resolve()
        return config
