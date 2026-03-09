from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from wsi_seg.slide import OpenSlideReader


@dataclass(slots=True)
class CoarseTissueMask:
    mask: np.ndarray
    integral: np.ndarray
    thumb_w: int
    thumb_h: int

    @classmethod
    def from_mask(cls, mask: np.ndarray) -> CoarseTissueMask:
        mask_u8 = mask.astype(np.uint8)
        integral = np.pad(mask_u8.astype(np.int32).cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
        h, w = mask_u8.shape
        return cls(mask=mask_u8, integral=integral, thumb_w=w, thumb_h=h)

    def region_fraction(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        out_w: int,
        out_h: int,
    ) -> float:
        tx1 = int(np.floor(x1 * self.thumb_w / out_w))
        ty1 = int(np.floor(y1 * self.thumb_h / out_h))
        tx2 = int(np.ceil(x2 * self.thumb_w / out_w))
        ty2 = int(np.ceil(y2 * self.thumb_h / out_h))
        tx1 = int(np.clip(tx1, 0, self.thumb_w))
        ty1 = int(np.clip(ty1, 0, self.thumb_h))
        tx2 = int(np.clip(tx2, 0, self.thumb_w))
        ty2 = int(np.clip(ty2, 0, self.thumb_h))
        if tx2 <= tx1 or ty2 <= ty1:
            return 0.0
        total = (
            self.integral[ty2, tx2]
            - self.integral[ty1, tx2]
            - self.integral[ty2, tx1]
            + self.integral[ty1, tx1]
        )
        area = float((tx2 - tx1) * (ty2 - ty1))
        return float(total) / area if area > 0 else 0.0


def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    if total == 0:
        return 0
    sum_total = float(np.dot(np.arange(256), hist))
    sum_back = 0.0
    weight_back = 0.0
    max_var = -1.0
    threshold = 0
    for idx in range(256):
        weight_back += hist[idx]
        if weight_back == 0:
            continue
        weight_fore = total - weight_back
        if weight_fore == 0:
            break
        sum_back += idx * hist[idx]
        mean_back = sum_back / weight_back
        mean_fore = (sum_total - sum_back) / weight_fore
        between = weight_back * weight_fore * (mean_back - mean_fore) ** 2
        if between > max_var:
            max_var = between
            threshold = idx
    return threshold


def _cleanup_mask(mask: np.ndarray) -> np.ndarray:
    img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    # close then open using Pillow max/min filters
    img = img.filter(ImageFilter.MaxFilter(size=5))
    img = img.filter(ImageFilter.MinFilter(size=5))
    img = img.filter(ImageFilter.MinFilter(size=3))
    img = img.filter(ImageFilter.MaxFilter(size=3))
    return (np.asarray(img, dtype=np.uint8) > 0).astype(np.uint8)


def build_coarse_tissue_mask(
    slide: OpenSlideReader,
    *,
    max_size: int,
    saturation_threshold: int = 18,
    white_threshold: int = 235,
) -> CoarseTissueMask:
    thumb = slide.thumbnail(max_size)
    hsv = np.asarray(thumb.convert("HSV"), dtype=np.uint8)
    sat = hsv[..., 1]
    val = hsv[..., 2]
    gray = np.asarray(thumb.convert("L"), dtype=np.uint8)

    otsu = otsu_threshold(gray)
    tissue = ((gray < otsu) | ((sat >= saturation_threshold) & (val < white_threshold))).astype(
        np.uint8
    )
    tissue = _cleanup_mask(tissue)
    return CoarseTissueMask.from_mask(tissue)
