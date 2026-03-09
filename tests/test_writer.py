from pathlib import Path

import numpy as np
import tifffile

from wsi_seg.writer import export_mask_ome_tiff, export_mask_tiff


def test_export_mask_tiff_embeds_resolution_tags(tmp_path: Path) -> None:
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    mask[100:200, 200:300] = 1
    out = export_mask_tiff(
        mask,
        tmp_path / "mask.tif",
        mpp_x=0.88,
        mpp_y=0.91,
        tile_size=256,
    )
    assert out.exists()
    with tifffile.TiffFile(out) as tif:
        page = tif.pages[0]
        assert page.tags["XResolution"].value is not None
        assert page.tags["YResolution"].value is not None
        assert page.tags["ResolutionUnit"].value is not None


def test_export_mask_ome_tiff_writes_ome_metadata_and_pyramid(tmp_path: Path) -> None:
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    mask[64:512, 64:512] = 1
    out = export_mask_ome_tiff(
        mask,
        tmp_path / "mask.ome.tif",
        mpp_x=0.88,
        mpp_y=0.88,
        tile_size=256,
        pyramid_min_size=256,
    )
    assert out.exists()
    with tifffile.TiffFile(out) as tif:
        ome = tif.ome_metadata or ""
        assert "PhysicalSizeX=\"0.88\"" in ome
        assert "PhysicalSizeY=\"0.88\"" in ome
        assert len(tif.pages[0].pages) >= 1