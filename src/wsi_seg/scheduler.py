from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from wsi_seg.geometry import PatchMeta, axis_positions, valid_crop_bounds
from wsi_seg.slide import OpenSlideReader
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
    out_w, out_h = slide.output_shape(target_mpp)
    if not use_bounds or slide.metadata.bounds is None:
        return out_w, out_h, ScheduleROI(0, 0, out_w, out_h)

    bounds = slide.metadata.bounds
    x1 = int(bounds.x * slide.metadata.mpp_x / target_mpp)
    y1 = int(bounds.y * slide.metadata.mpp_y / target_mpp)
    x2 = int((bounds.x + bounds.width) * slide.metadata.mpp_x / target_mpp)
    y2 = int((bounds.y + bounds.height) * slide.metadata.mpp_y / target_mpp)
    x1 = max(0, min(out_w, x1))
    y1 = max(0, min(out_h, y1))
    x2 = max(x1, min(out_w, x2))
    y2 = max(y1, min(out_h, y2))
    return out_w, out_h, ScheduleROI(x1, y1, x2 - x1, y2 - y1)


def _crop_intersects_roi(
    crop_x1: int, crop_y1: int, crop_x2: int, crop_y2: int, roi: ScheduleROI
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
                tissue_fraction = coarse_mask.region_fraction(gx1, gy1, gx2, gy2, out_w, out_h)
                if tissue_fraction < min_tissue_fraction:
                    continue
            tissue_patches += 1
            is_edge = (
                out_x == 0
                or out_y == 0
                or out_x + patch_px >= out_w
                or out_y + patch_px >= out_h
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
