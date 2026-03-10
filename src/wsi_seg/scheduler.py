from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from wsi_seg.geometry import PatchMeta, axis_positions, valid_crop_bounds
from wsi_seg.slide import OpenSlideReader, OutputFrame
from wsi_seg.tissue import CoarseTissueMask


@dataclass(slots=True)
class ScheduleROI:
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height


@dataclass(slots=True)
class SuperTilePlan:
    out_x: int
    out_y: int
    out_w: int
    out_h: int
    patches: list[PatchMeta]


@dataclass(slots=True)
class PlanningSummary:
    roi: ScheduleROI
    total_grid_patches: int
    roi_patches: int
    tissue_patches: int
    supertiles: int
    edge_patches: int = 0
    padded_patches: int = 0


def schedule_roi(
    slide: OpenSlideReader, target_mpp: float, use_bounds: bool
) -> tuple[int, int, ScheduleROI]:
    # Once an output frame is selected, scheduling becomes local to that frame.
    # Bounds handling is therefore resolved up front instead of scattered
    # throughout patch planning.
    frame = slide.output_frame(target_mpp, use_bounds)
    return frame.out_w, frame.out_h, ScheduleROI(0, 0, frame.out_w, frame.out_h)


def _crop_intersects_roi(
    crop_x1: int,
    crop_y1: int,
    crop_x2: int,
    crop_y2: int,
    roi: ScheduleROI,
) -> bool:
    return not (crop_x2 <= roi.x or crop_y2 <= roi.y or crop_x1 >= roi.x2 or crop_y1 >= roi.y2)


def plan_patch_grid(
    *,
    out_w: int,
    out_h: int,
    patch_px: int,
    stride_px: int,
    halo_px: int,
    roi: ScheduleROI,
    coarse_mask: CoarseTissueMask | None,
    min_tissue_fraction: float,
    frame: OutputFrame | None = None,
    slide: OpenSlideReader | None = None,
) -> tuple[list[PatchMeta], PlanningSummary]:
    xs = axis_positions(out_w, patch_px, stride_px)
    ys = axis_positions(out_h, patch_px, stride_px)
    total_grid = len(xs) * len(ys)
    roi_patches = 0
    tissue_patches = 0
    edge_patches = 0
    padded_patches = 0
    metas: list[PatchMeta] = []

    for out_y in ys:
        for out_x in xs:
            left, top, right, bottom = valid_crop_bounds(
                out_x, out_y, patch_px, out_w, out_h, halo_px
            )
            gx1 = out_x + left
            gy1 = out_y + top
            gx2 = min(out_x + right, out_w)
            gy2 = min(out_y + bottom, out_h)
            if not _crop_intersects_roi(gx1, gy1, gx2, gy2, roi):
                continue
            roi_patches += 1
            if coarse_mask is not None:
                if frame is not None and slide is not None:
                    # The tissue mask was computed on a full-slide thumbnail, so
                    # frame-local output coordinates must be projected back to
                    # global level-0 coordinates before tissue gating.
                    l0_x1 = frame.origin_x_level0 + int(
                        math.floor(gx1 * frame.actual_output_mpp_x / slide.metadata.mpp_x)
                    )
                    l0_y1 = frame.origin_y_level0 + int(
                        math.floor(gy1 * frame.actual_output_mpp_y / slide.metadata.mpp_y)
                    )
                    l0_x2 = frame.origin_x_level0 + int(
                        math.ceil(gx2 * frame.actual_output_mpp_x / slide.metadata.mpp_x)
                    )
                    l0_y2 = frame.origin_y_level0 + int(
                        math.ceil(gy2 * frame.actual_output_mpp_y / slide.metadata.mpp_y)
                    )
                    tissue_fraction = coarse_mask.region_fraction_level0(
                        l0_x1,
                        l0_y1,
                        l0_x2,
                        l0_y2,
                        slide_width_level0=slide.metadata.width,
                        slide_height_level0=slide.metadata.height,
                    )
                else:
                    tissue_fraction = coarse_mask.region_fraction(
                        gx1,
                        gy1,
                        gx2,
                        gy2,
                        out_w,
                        out_h,
                    )
                if tissue_fraction < min_tissue_fraction:
                    continue
            tissue_patches += 1
            is_edge = (
                out_x == 0 or out_y == 0 or out_x + patch_px >= out_w or out_y + patch_px >= out_h
            )
            is_padded = out_x + patch_px > out_w or out_y + patch_px > out_h
            if is_edge:
                edge_patches += 1
            if is_padded:
                padded_patches += 1
            metas.append(PatchMeta(out_x=out_x, out_y=out_y))

    summary = PlanningSummary(
        roi=roi,
        total_grid_patches=total_grid,
        roi_patches=roi_patches,
        tissue_patches=tissue_patches,
        supertiles=0,
        edge_patches=edge_patches,
        padded_patches=padded_patches,
    )
    return metas, summary


def group_patches_into_supertiles(
    metas: Iterable[PatchMeta],
    *,
    supertile_px: int,
    patch_px: int,
) -> list[SuperTilePlan]:
    groups: dict[tuple[int, int], list[PatchMeta]] = defaultdict(list)
    for meta in metas:
        key = (meta.out_x // supertile_px, meta.out_y // supertile_px)
        groups[key].append(meta)

    plans: list[SuperTilePlan] = []
    for _, group in sorted(groups.items(), key=lambda item: item[0]):
        min_x = min(meta.out_x for meta in group)
        min_y = min(meta.out_y for meta in group)
        max_x = max(meta.out_x for meta in group) + patch_px
        max_y = max(meta.out_y for meta in group) + patch_px
        plans.append(
            SuperTilePlan(
                out_x=min_x,
                out_y=min_y,
                out_w=max_x - min_x,
                out_h=max_y - min_y,
                patches=sorted(group, key=lambda m: (m.out_y, m.out_x)),
            )
        )
    return plans
