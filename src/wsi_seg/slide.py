from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(slots=True)
class SlideLevel:
    index: int
    width: int
    height: int
    downsample: float
    mpp_x: float
    mpp_y: float


@dataclass(slots=True)
class SlideBounds:
    x: int
    y: int
    width: int
    height: int


@dataclass(slots=True)
class SlideMetadata:
    path: Path
    vendor: str | None
    width: int
    height: int
    mpp_x: float
    mpp_y: float
    objective_power: float | None
    levels: list[SlideLevel]
    bounds: SlideBounds | None
    background_hex: str | None


@dataclass(slots=True)
class LevelSelection:
    level: int
    downsample: float
    level_mpp_x: float
    level_mpp_y: float
    error_x: float
    error_y: float


class OpenSlideReader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._openslide = self._import_openslide()
        self._slide = self._openslide.OpenSlide(str(self.path))
        self.metadata = self._read_metadata()

    @staticmethod
    def _import_openslide():
        try:
            import openslide  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "OpenSlide is not available. Install openslide-python and the native OpenSlide runtime."  # noqa: E501
            ) from exc
        return openslide

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _read_metadata(self) -> SlideMetadata:
        osr = self._slide
        osl = self._openslide
        props = dict(osr.properties)

        mpp_x = self._float_or_none(props.get(osl.PROPERTY_NAME_MPP_X))
        mpp_y = self._float_or_none(props.get(osl.PROPERTY_NAME_MPP_Y))
        if mpp_x is None or mpp_y is None:
            aperio_mpp = self._float_or_none(props.get("aperio.MPP"))
            if aperio_mpp is not None:
                mpp_x = mpp_x or aperio_mpp
                mpp_y = mpp_y or aperio_mpp
        if mpp_x is None or mpp_y is None:
            raise ValueError("Could not determine level-0 MPP from slide metadata.")

        objective_power = self._float_or_none(props.get(osl.PROPERTY_NAME_OBJECTIVE_POWER))
        background_hex = props.get(osl.PROPERTY_NAME_BACKGROUND_COLOR)
        vendor = props.get(osl.PROPERTY_NAME_VENDOR)

        bounds = None
        bx = self._float_or_none(props.get(osl.PROPERTY_NAME_BOUNDS_X))
        by = self._float_or_none(props.get(osl.PROPERTY_NAME_BOUNDS_Y))
        bw = self._float_or_none(props.get(osl.PROPERTY_NAME_BOUNDS_WIDTH))
        bh = self._float_or_none(props.get(osl.PROPERTY_NAME_BOUNDS_HEIGHT))
        if None not in (bx, by, bw, bh):
            bounds = SlideBounds(int(bx), int(by), int(bw), int(bh))

        levels: list[SlideLevel] = []
        dims_ds = zip(osr.level_dimensions, osr.level_downsamples, strict=True)
        for idx, ((w, h), ds) in enumerate(dims_ds):
            levels.append(
                SlideLevel(
                    index=idx,
                    width=int(w),
                    height=int(h),
                    downsample=float(ds),
                    mpp_x=float(mpp_x * ds),
                    mpp_y=float(mpp_y * ds),
                )
            )

        w0, h0 = osr.dimensions
        return SlideMetadata(
            path=self.path,
            vendor=vendor,
            width=int(w0),
            height=int(h0),
            mpp_x=float(mpp_x),
            mpp_y=float(mpp_y),
            objective_power=objective_power,
            levels=levels,
            bounds=bounds,
            background_hex=background_hex,
        )

    def set_cache(self, capacity_bytes: int) -> bool:
        if capacity_bytes <= 0:
            return False
        try:
            cache = self._openslide.OpenSlideCache(capacity_bytes)
            self._slide.set_cache(cache)
            return True
        except Exception:
            return False

    def choose_level(self, target_mpp: float) -> LevelSelection:
        best: SlideLevel | None = None
        best_score: float | None = None
        for level in self.metadata.levels:
            score = max(abs(level.mpp_x - target_mpp), abs(level.mpp_y - target_mpp))
            if best is None or best_score is None or score < best_score:
                best = level
                best_score = score
        assert best is not None
        return LevelSelection(
            level=best.index,
            downsample=best.downsample,
            level_mpp_x=best.mpp_x,
            level_mpp_y=best.mpp_y,
            error_x=abs(best.mpp_x - target_mpp),
            error_y=abs(best.mpp_y - target_mpp),
        )

    def output_shape(self, target_mpp: float) -> tuple[int, int]:
        out_w = max(1, int(round(self.metadata.width * self.metadata.mpp_x / target_mpp)))
        out_h = max(1, int(round(self.metadata.height * self.metadata.mpp_y / target_mpp)))
        return out_w, out_h

    def _background_rgba(self) -> tuple[int, int, int, int]:
        text = self.metadata.background_hex
        if text is None:
            return (255, 255, 255, 255)
        text = text.lstrip("#")
        if len(text) != 6:
            return (255, 255, 255, 255)
        try:
            return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16), 255)
        except ValueError:
            return (255, 255, 255, 255)

    def read_region_rgb(self, location0: tuple[int, int], level: int, size: tuple[int, int]):
        rgba = self._slide.read_region(location0, level, size)
        bg = Image.new("RGBA", rgba.size, self._background_rgba())
        rgb = Image.alpha_composite(bg, rgba).convert("RGB")
        return rgb

    def thumbnail(self, max_size: int) -> Image.Image:
        return self._slide.get_thumbnail((max_size, max_size)).convert("RGB")

    def close(self) -> None:
        self._slide.close()

    def __enter__(self) -> OpenSlideReader:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False
