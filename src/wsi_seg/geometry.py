from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PatchMeta:
    out_x: int
    out_y: int


def axis_positions(length: int, patch_px: int, stride_px: int) -> list[int]:
    if length <= patch_px:
        return [0]
    positions = list(range(0, length - patch_px + 1, stride_px))
    last = length - patch_px
    if positions[-1] != last:
        positions.append(last)
    return positions


def valid_crop_bounds(
    out_x: int,
    out_y: int,
    patch_px: int,
    out_w: int,
    out_h: int,
    halo_px: int,
) -> tuple[int, int, int, int]:
    left = 0 if out_x == 0 else halo_px
    top = 0 if out_y == 0 else halo_px
    right = patch_px if out_x + patch_px >= out_w else patch_px - halo_px
    bottom = patch_px if out_y + patch_px >= out_h else patch_px - halo_px
    return left, top, right, bottom
