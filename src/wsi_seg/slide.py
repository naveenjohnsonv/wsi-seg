from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
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
    backend: str
    detected_format: str | None
    width: int
    height: int
    mpp_x: float
    mpp_y: float
    mpp_source: str
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
    policy: str
    resize_factor_x: float
    resize_factor_y: float


class OpenSlideReader:
    _TEXT_MPP_PATTERNS = (
        r"\bmpp(?:[_\s-]*x)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        r"\bmicrons?\s+per\s+pixel\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)",
        r"\bmicrometers?\s+per\s+pixel\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)",
        r"\b(?:pixel\s+size|pixelsize)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(?:um|µm)",
    )

    def __init__(
        self,
        path: str | Path,
        *,
        mpp_override_x: float | None = None,
        mpp_override_y: float | None = None,
    ) -> None:
        self.path = Path(path)
        self._openslide = self._import_openslide()
        self._mpp_override_x = mpp_override_x
        self._mpp_override_y = mpp_override_y
        self._slide = self._openslide.open_slide(str(self.path))
        self.metadata = self._read_metadata()

    @staticmethod
    def _import_openslide():
        try:
            import openslide  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "OpenSlide is not available. "
                "Install openslide-python and the native OpenSlide runtime."
            ) from exc
        return openslide

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            pass
        frac = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)\s*$", text)
        if frac:
            num = float(frac.group(1))
            den = float(frac.group(2))
            if den != 0:
                return num / den
        return None

    @classmethod
    def _parse_mpp_from_text(cls, value: Any) -> float | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        for pattern in cls._TEXT_MPP_PATTERNS:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return cls._float_or_none(match.group(1))
        return None

    @classmethod
    def _extract_mpp_from_properties(
        cls,
        props: dict[str, Any],
        *,
        property_name_mpp_x: str = "openslide.mpp-x",
        property_name_mpp_y: str = "openslide.mpp-y",
        property_name_objective_power: str = "openslide.objective-power",
    ) -> tuple[float, float, str]:
        normalized = {str(k).lower(): v for k, v in props.items()}

        def first_float(keys: tuple[str, ...]) -> float | None:
            for key in keys:
                val = normalized.get(key.lower())
                parsed = cls._float_or_none(val)
                if parsed is not None and parsed > 0:
                    return parsed
            return None

        mpp_x = first_float((property_name_mpp_x, "openslide.mpp-x", "mpp_x", "mppx"))
        mpp_y = first_float((property_name_mpp_y, "openslide.mpp-y", "mpp_y", "mppy"))
        if mpp_x is not None or mpp_y is not None:
            if mpp_x is None:
                mpp_x = mpp_y
            if mpp_y is None:
                mpp_y = mpp_x
            assert mpp_x is not None and mpp_y is not None
            return float(mpp_x), float(mpp_y), "openslide.mpp-x/y"

        aperio_mpp = first_float(("aperio.mpp",))
        if aperio_mpp is not None:
            return float(aperio_mpp), float(aperio_mpp), "aperio.MPP"

        for text_key in ("openslide.comment", "tiff.imagedescription", "image_description"):
            parsed = cls._parse_mpp_from_text(normalized.get(text_key))
            if parsed is not None and parsed > 0:
                return float(parsed), float(parsed), f"text:{text_key}"

        x_res = cls._float_or_none(normalized.get("tiff.xresolution"))
        y_res = cls._float_or_none(normalized.get("tiff.yresolution"))
        unit = str(normalized.get("tiff.resolutionunit", "")).strip().lower()
        um_per_unit = None
        if unit in {"inch", "inches", "2"}:
            um_per_unit = 25400.0
        elif unit in {"centimeter", "centimetre", "cm", "3"}:
            um_per_unit = 10000.0
        if um_per_unit is not None and (x_res or y_res):
            if x_res is None:
                x_res = y_res
            if y_res is None:
                y_res = x_res
            assert x_res is not None and y_res is not None
            if x_res > 0 and y_res > 0:
                return (
                    float(um_per_unit / x_res),
                    float(um_per_unit / y_res),
                    "tiff.resolution",
                )

        objective_power = first_float(
            (
                property_name_objective_power,
                "aperio.appmag",
                "hamamatsu.sourcelens",
                "objective_power",
            )
        )
        if objective_power is not None and objective_power > 0:
            approx = 10.0 / float(objective_power)
            return approx, approx, "objective-power-estimate"

        raise ValueError("Could not determine level-0 MPP from slide metadata.")


    @classmethod
    def _override_mpp(
        cls,
        override_x: float | None,
        override_y: float | None,
    ) -> tuple[float, float, str] | None:
        if override_x is None and override_y is None:
            return None
        if override_x is None:
            override_x = override_y
        if override_y is None:
            override_y = override_x
        if override_x is None or override_y is None or override_x <= 0 or override_y <= 0:
            raise ValueError("MPP overrides must be > 0 when provided")
        return float(override_x), float(override_y), "config.override"

    def _resolve_mpp(self, props: dict[str, Any], osl: Any) -> tuple[float, float, str]:
        override = self._override_mpp(self._mpp_override_x, self._mpp_override_y)
        if override is not None:
            return override
        return self._extract_mpp_from_properties(
            props,
            property_name_mpp_x=osl.PROPERTY_NAME_MPP_X,
            property_name_mpp_y=osl.PROPERTY_NAME_MPP_Y,
            property_name_objective_power=osl.PROPERTY_NAME_OBJECTIVE_POWER,
        )

    def _detect_format(self) -> str | None:
        detect_format = getattr(self._openslide, "detect_format", None)
        if callable(detect_format):
            try:
                detected = detect_format(str(self.path))
            except Exception:  # pragma: no cover
                return None
            return str(detected) if detected is not None else None
        return None

    def _read_metadata(self) -> SlideMetadata:
        osr = self._slide
        osl = self._openslide
        props = dict(getattr(osr, "properties", {}))
        backend = type(osr).__name__
        detected_format = self._detect_format()

        mpp_x, mpp_y, mpp_source = self._resolve_mpp(props, osl)

        objective_power = self._float_or_none(props.get(osl.PROPERTY_NAME_OBJECTIVE_POWER))
        background_hex = props.get(osl.PROPERTY_NAME_BACKGROUND_COLOR)
        vendor = props.get(osl.PROPERTY_NAME_VENDOR)
        if vendor is None and detected_format is not None:
            vendor = detected_format
        if vendor is None and backend == "ImageSlide":
            vendor = "generic-image"

        bounds = None
        bx = self._float_or_none(props.get(osl.PROPERTY_NAME_BOUNDS_X))
        by = self._float_or_none(props.get(osl.PROPERTY_NAME_BOUNDS_Y))
        bw = self._float_or_none(props.get(osl.PROPERTY_NAME_BOUNDS_WIDTH))
        bh = self._float_or_none(props.get(osl.PROPERTY_NAME_BOUNDS_HEIGHT))
        if None not in (bx, by, bw, bh):
            bounds = SlideBounds(int(bx), int(by), int(bw), int(bh))

        levels: list[SlideLevel] = []
        level_dimensions = getattr(osr, "level_dimensions", [osr.dimensions])
        level_downsamples = getattr(osr, "level_downsamples", [1.0])
        dims_ds = zip(level_dimensions, level_downsamples, strict=True)
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
            backend=backend,
            detected_format=detected_format,
            width=int(w0),
            height=int(h0),
            mpp_x=float(mpp_x),
            mpp_y=float(mpp_y),
            mpp_source=mpp_source,
            objective_power=objective_power,
            levels=levels,
            bounds=bounds,
            background_hex=background_hex,
        )

    def set_cache(self, capacity_bytes: int) -> bool:
        if capacity_bytes <= 0:
            return False
        if not hasattr(self._openslide, "OpenSlideCache") or not hasattr(self._slide, "set_cache"):
            return False
        try:
            cache = self._openslide.OpenSlideCache(capacity_bytes)
            self._slide.set_cache(cache)
            return True
        except Exception:
            return False

    def _nearest_level(self, target_mpp: float) -> SlideLevel:
        best: SlideLevel | None = None
        best_score: float | None = None
        for level in self.metadata.levels:
            score = max(abs(level.mpp_x - target_mpp), abs(level.mpp_y - target_mpp))
            if best is None or best_score is None or score < best_score:
                best = level
                best_score = score
        assert best is not None
        return best

    @staticmethod
    def _native_oversample_factor(level: SlideLevel, target_mpp: float) -> float:
        return max(target_mpp / level.mpp_x, target_mpp / level.mpp_y)

    def _prefer_higher_level(
        self,
        target_mpp: float,
        *,
        max_native_oversample_factor: float | None = None,
    ) -> SlideLevel | None:
        candidates = [
            level
            for level in self.metadata.levels
            if level.mpp_x <= target_mpp and level.mpp_y <= target_mpp
        ]
        candidates.sort(key=lambda level: (level.mpp_x + level.mpp_y) / 2.0, reverse=True)
        for level in candidates:
            if max_native_oversample_factor is None:
                return level
            if self._native_oversample_factor(level, target_mpp) <= max_native_oversample_factor:
                return level
        return None

    def choose_level(
        self,
        target_mpp: float,
        *,
        policy: str = "nearest",
        max_native_oversample_factor: float = 2.0,
    ) -> LevelSelection:
        if policy == "nearest":
            best = self._nearest_level(target_mpp)
        elif policy == "prefer_higher":
            best = self._prefer_higher_level(target_mpp) or self._nearest_level(target_mpp)
        elif policy == "prefer_higher_bounded":
            best = self._prefer_higher_level(
                target_mpp,
                max_native_oversample_factor=max_native_oversample_factor,
            ) or self._nearest_level(target_mpp)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported level-selection policy: {policy}")

        return LevelSelection(
            level=best.index,
            downsample=best.downsample,
            level_mpp_x=best.mpp_x,
            level_mpp_y=best.mpp_y,
            error_x=abs(best.mpp_x - target_mpp),
            error_y=abs(best.mpp_y - target_mpp),
            policy=policy,
            resize_factor_x=target_mpp / best.mpp_x,
            resize_factor_y=target_mpp / best.mpp_y,
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

    def read_region_rgb(
        self,
        location0: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> Image.Image:
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


def read_output_region(
    slide: OpenSlideReader,
    selection: LevelSelection,
    *,
    out_x: int,
    out_y: int,
    out_w: int,
    out_h: int,
    target_mpp: float,
) -> np.ndarray:
    mpp_x = slide.metadata.mpp_x
    mpp_y = slide.metadata.mpp_y
    level_mpp_x = selection.level_mpp_x
    level_mpp_y = selection.level_mpp_y

    read_x = int(round(out_x * target_mpp / mpp_x))
    read_y = int(round(out_y * target_mpp / mpp_y))
    read_w = max(1, int(round(out_w * target_mpp / level_mpp_x)))
    read_h = max(1, int(round(out_h * target_mpp / level_mpp_y)))

    pil_img = slide.read_region_rgb((read_x, read_y), selection.level, (read_w, read_h))
    if pil_img.size != (out_w, out_h):
        pil_img = pil_img.resize((out_w, out_h), resample=Image.Resampling.BILINEAR)
    return np.asarray(pil_img, dtype=np.uint8)
