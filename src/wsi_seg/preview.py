from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from wsi_seg.slide import OpenSlideReader, OutputFrame


def _overlay_mask(
    base: Image.Image,
    mask_img: Image.Image,
    rgba: tuple[int, int, int, int],
) -> Image.Image:
    overlay = base.convert("RGBA")
    layer = Image.new("RGBA", base.size, rgba[:3] + (0,))
    alpha = mask_img.point(lambda p: rgba[3] if p > 0 else 0).convert("L")
    layer.putalpha(alpha)
    return Image.alpha_composite(overlay, layer)


def _mask_to_thumb(
    mask: np.ndarray,
    thumb_size: tuple[int, int],
    *,
    slide: OpenSlideReader,
    frame: OutputFrame | None,
) -> Image.Image:
    # Previews are rendered on a full-slide thumbnail, so a bounds-local mask
    # must be resized into the thumbnail region that corresponds to the output
    # frame's native level-0 extent.
    mask_img = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, mode="L")
    if frame is None:
        return mask_img.resize(thumb_size, resample=Image.Resampling.NEAREST)

    thumb_w, thumb_h = thumb_size
    full_w = slide.metadata.width
    full_h = slide.metadata.height
    paste_x = int(round(frame.origin_x_level0 * thumb_w / full_w))
    paste_y = int(round(frame.origin_y_level0 * thumb_h / full_h))
    paste_w = max(1, int(round(frame.width_level0 * thumb_w / full_w)))
    paste_h = max(1, int(round(frame.height_level0 * thumb_h / full_h)))

    resized = mask_img.resize((paste_w, paste_h), resample=Image.Resampling.NEAREST)
    canvas = Image.new("L", thumb_size, 0)
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def save_previews(
    slide: OpenSlideReader,
    mask: np.ndarray,
    out_dir: str | Path,
    *,
    max_size: int = 1536,
    tissue_mask: np.ndarray | None = None,
    frame: OutputFrame | None = None,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    thumb = slide.thumbnail(max_size)
    result: dict[str, Path] = {}

    mask_thumb = _mask_to_thumb(mask, thumb.size, slide=slide, frame=frame)
    bw_path = out / "preview_mask.png"
    mask_thumb.save(bw_path)
    result["preview_mask"] = bw_path

    overlay = _overlay_mask(thumb, mask_thumb, (0, 255, 0, 96))
    overlay_path = out / "preview_overlay.png"
    overlay.save(overlay_path)
    result["preview_overlay"] = overlay_path

    if tissue_mask is not None:
        tissue_img = Image.fromarray((np.asarray(tissue_mask) > 0).astype(np.uint8) * 255, mode="L")
        tissue_thumb = tissue_img.resize(thumb.size, resample=Image.Resampling.NEAREST)
        tissue_path = out / "preview_tissue.png"
        tissue_thumb.save(tissue_path)
        result["preview_tissue"] = tissue_path

    return result
