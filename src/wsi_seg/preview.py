from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from wsi_seg.slide import OpenSlideReader


def save_previews(
    slide: OpenSlideReader,
    mask: np.ndarray,
    out_dir: str | Path,
    *,
    max_size: int = 1536,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    thumb = slide.thumbnail(max_size)
    mask_img = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, mode="L")
    mask_thumb = mask_img.resize(thumb.size, resample=Image.Resampling.NEAREST)

    bw_path = out / "preview_mask.png"
    mask_thumb.save(bw_path)

    thumb_rgba = thumb.convert("RGBA")
    green = Image.new("RGBA", thumb.size, (0, 255, 0, 0))
    alpha = mask_thumb.point(lambda p: 96 if p > 0 else 0).convert("L")
    green.putalpha(alpha)
    overlay = Image.alpha_composite(thumb_rgba, green)

    overlay_path = out / "preview_overlay.png"
    overlay.save(overlay_path)

    return {"preview_mask": bw_path, "preview_overlay": overlay_path}
