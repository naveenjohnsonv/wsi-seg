from types import SimpleNamespace

import numpy as np
from PIL import Image

from wsi_seg.scheduler import (
    ScheduleROI,
    group_patches_into_supertiles,
    plan_patch_grid,
    schedule_roi,
)
from wsi_seg.tissue import build_coarse_tissue_mask, otsu_threshold


class _DummySlide:
    def __init__(self) -> None:
        self.metadata = SimpleNamespace(
            width=1000,
            height=800,
            mpp_x=0.5,
            mpp_y=0.5,
            bounds=SimpleNamespace(x=200, y=100, width=400, height=300),
        )

    def output_shape(self, target_mpp: float) -> tuple[int, int]:
        out_w = round(self.metadata.width * self.metadata.mpp_x / target_mpp)
        out_h = round(self.metadata.height * self.metadata.mpp_y / target_mpp)
        return out_w, out_h

    def output_frame(self, target_mpp: float, use_bounds: bool):
        if use_bounds:
            return SimpleNamespace(out_w=200, out_h=150)
        return SimpleNamespace(out_w=500, out_h=400)


def test_schedule_roi_uses_bounds_projection() -> None:
    slide = _DummySlide()
    out_w, out_h, roi = schedule_roi(slide, target_mpp=1.0, use_bounds=True)
    assert (out_w, out_h) == (200, 150)
    assert (roi.x, roi.y, roi.width, roi.height) == (0, 0, 200, 150)


def test_plan_patch_grid_filters_by_roi_and_mask() -> None:
    coarse = np.zeros((10, 10), dtype=np.uint8)
    coarse[:, :5] = 1
    from wsi_seg.tissue import CoarseTissueMask

    coarse_mask = CoarseTissueMask.from_mask(coarse)
    metas, summary = plan_patch_grid(
        out_w=1024,
        out_h=1024,
        patch_px=512,
        stride_px=384,
        halo_px=64,
        roi=ScheduleROI(0, 0, 1024, 1024),
        coarse_mask=coarse_mask,
        min_tissue_fraction=0.2,
    )
    assert summary.total_grid_patches == 9
    assert summary.roi_patches == 9
    assert 0 < len(metas) < 9


def test_group_patches_into_supertiles_groups_nearby_positions() -> None:
    from wsi_seg.geometry import PatchMeta

    plans = group_patches_into_supertiles(
        [
            PatchMeta(0, 0),
            PatchMeta(384, 0),
            PatchMeta(4608, 0),
        ],
        supertile_px=4096,
        patch_px=512,
    )
    assert len(plans) == 2
    assert len(plans[0].patches) == 2
    assert len(plans[1].patches) == 1


class _ThumbSlide:
    def thumbnail(self, max_size: int) -> Image.Image:
        arr = np.full((128, 128, 3), 255, dtype=np.uint8)
        arr[32:96, 16:80] = np.array([180, 60, 120], dtype=np.uint8)
        return Image.fromarray(arr)


def test_otsu_threshold_returns_byte_threshold() -> None:
    gray = np.concatenate(
        [
            np.zeros((16, 16), dtype=np.uint8) + 30,
            np.zeros((16, 16), dtype=np.uint8) + 220,
        ],
        axis=1,
    )
    t = otsu_threshold(gray)
    assert 0 <= t <= 255


def test_build_coarse_tissue_mask_detects_colored_region() -> None:
    coarse = build_coarse_tissue_mask(_ThumbSlide(), max_size=128)
    assert coarse.mask.mean() > 0.05
