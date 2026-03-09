from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from numpy.lib.format import open_memmap


def create_mask_memmap(path: str | Path, shape: tuple[int, int]) -> np.memmap:
    mm = open_memmap(str(path), mode="w+", dtype=np.uint8, shape=shape)
    mm[:] = 0
    return mm


def _pixels_per_centimeter(mpp: float) -> float:
    return 10000.0 / float(mpp)


def _count_pyramid_levels(shape: tuple[int, ...], min_size: int) -> int:
    count = 0
    h, w = shape[0], shape[1]
    effective_min = max(2, min_size)
    while min(h, w) >= effective_min:
        nh, nw = h // 2, w // 2
        if nh == h and nw == w:
            break
        count += 1
        h, w = nh, nw
    return count


def _iter_pyramid_levels(mask: np.ndarray, min_size: int) -> Iterator[np.ndarray]:
    current = np.asarray(mask, dtype=np.uint8)
    while min(current.shape) >= max(2, min_size):
        next_level = np.ascontiguousarray(current[::2, ::2])
        if next_level.shape == current.shape:
            break
        yield next_level
        current = next_level


def _tile_option(shape: tuple[int, int], tile_size: int) -> tuple[int, int] | None:
    if tile_size <= 0:
        return None
    if shape[0] < tile_size or shape[1] < tile_size:
        return None
    return (tile_size, tile_size)


def export_mask_tiff(
    mask: np.ndarray,
    path: str | Path,
    *,
    mpp_x: float,
    mpp_y: float,
    bigtiff: bool = True,
    compression: str | None = "zlib",
    tile_size: int = 512,
    description: dict[str, Any] | None = None,
) -> Path:
    out = Path(path)
    payload = None if description is None else json.dumps(description, sort_keys=True)
    options: dict[str, object] = {
        "photometric": "minisblack",
        "bigtiff": bigtiff,
        "compression": compression,
        "resolution": (_pixels_per_centimeter(mpp_x), _pixels_per_centimeter(mpp_y)),
        "resolutionunit": "CENTIMETER",
        "software": "wsi-segmentation-pipeline",
        "description": payload,
        "metadata": None,
    }
    tile = _tile_option(np.asarray(mask).shape, tile_size)
    if tile is not None:
        options["tile"] = tile
    tifffile.imwrite(out, np.asarray(mask, dtype=np.uint8), **options)
    return out


def export_mask_ome_tiff(
    mask: np.ndarray,
    path: str | Path,
    *,
    mpp_x: float,
    mpp_y: float,
    bigtiff: bool = True,
    compression: str | None = "zlib",
    tile_size: int = 512,
    pyramid_min_size: int = 512,
) -> Path:
    out = Path(path)
    base = np.asarray(mask, dtype=np.uint8)
    num_levels = _count_pyramid_levels(base.shape, pyramid_min_size)
    ome_meta = {
        "axes": "YX",
        "PhysicalSizeX": float(mpp_x),
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": float(mpp_y),
        "PhysicalSizeYUnit": "µm",
        "Name": out.stem,
    }

    with tifffile.TiffWriter(out, bigtiff=bigtiff, ome=True) as tif:
        tile = _tile_option(base.shape, tile_size)
        base_options: dict[str, object] = {
            "photometric": "minisblack",
            "compression": compression,
            "metadata": ome_meta,
            "resolution": (_pixels_per_centimeter(mpp_x), _pixels_per_centimeter(mpp_y)),
            "resolutionunit": "CENTIMETER",
            "subifds": num_levels,
        }
        if tile is not None:
            base_options["tile"] = tile
        tif.write(base, dtype=np.uint8, **base_options)

        level_mpp_x = float(mpp_x)
        level_mpp_y = float(mpp_y)
        for level in _iter_pyramid_levels(base, pyramid_min_size):
            level_mpp_x *= 2.0
            level_mpp_y *= 2.0
            level_tile = _tile_option(level.shape, tile_size)
            level_options: dict[str, object] = {
                "photometric": "minisblack",
                "compression": compression,
                "metadata": None,
                "subfiletype": 1,
                "resolution": (
                    _pixels_per_centimeter(level_mpp_x),
                    _pixels_per_centimeter(level_mpp_y),
                ),
                "resolutionunit": "CENTIMETER",
            }
            if level_tile is not None:
                level_options["tile"] = level_tile
            tif.write(np.asarray(level, dtype=np.uint8), **level_options)
    return out
