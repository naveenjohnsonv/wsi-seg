from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from numpy.lib.format import open_memmap


def create_mask_memmap(path: str | Path, shape: tuple[int, int]) -> np.memmap:
    mm = open_memmap(str(path), mode="w+", dtype=np.uint8, shape=shape)
    mm[:] = 0
    return mm


def export_mask_tiff(
    mask: np.ndarray,
    path: str | Path,
    *,
    bigtiff: bool = True,
    compression: str | None = None,
) -> Path:
    out = Path(path)
    tifffile.imwrite(
        out,
        np.asarray(mask, dtype=np.uint8),
        photometric="minisblack",
        bigtiff=bigtiff,
        compression=compression,
    )
    return out
