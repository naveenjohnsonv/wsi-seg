from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from wsi_seg.config import AppConfig
from wsi_seg.model import batch_to_tensor, load_torchscript_model, output_to_probs
from wsi_seg.preview import save_previews
from wsi_seg.slide import LevelSelection, OpenSlideReader
from wsi_seg.utils import dump_json, resolve_device, supports_amp
from wsi_seg.writer import create_mask_memmap, export_mask_tiff


@dataclass(slots=True)
class PatchMeta:
    out_x: int
    out_y: int


@dataclass(slots=True)
class RunArtifacts:
    mask_memmap: Path
    mask_tiff: Path | None
    preview_overlay: Path | None
    preview_mask: Path | None
    run_json: Path


@dataclass(slots=True)
class RunSummary:
    slide_path: Path
    output_shape: tuple[int, int]
    num_patches: int
    device: str
    total_seconds: float
    artifacts: RunArtifacts


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


def _resize_patch(pil_img: Image.Image, patch_px: int) -> np.ndarray:
    if pil_img.size != (patch_px, patch_px):
        pil_img = pil_img.resize((patch_px, patch_px), resample=Image.Resampling.BILINEAR)
    return np.asarray(pil_img, dtype=np.uint8)


def _read_patch(
    slide: OpenSlideReader,
    selection: LevelSelection,
    *,
    out_x: int,
    out_y: int,
    target_mpp: float,
    patch_px: int,
) -> np.ndarray:
    mpp_x = slide.metadata.mpp_x
    mpp_y = slide.metadata.mpp_y
    level_mpp_x = selection.level_mpp_x
    level_mpp_y = selection.level_mpp_y

    read_x = int(round(out_x * target_mpp / mpp_x))
    read_y = int(round(out_y * target_mpp / mpp_y))
    read_w = max(1, int(round(patch_px * target_mpp / level_mpp_x)))
    read_h = max(1, int(round(patch_px * target_mpp / level_mpp_y)))

    pil_img = slide.read_region_rgb((read_x, read_y), selection.level, (read_w, read_h))
    return _resize_patch(pil_img, patch_px)


def _write_patch(
    mask: np.ndarray,
    probs: np.ndarray,
    meta: PatchMeta,
    *,
    patch_px: int,
    halo_px: int,
    out_w: int,
    out_h: int,
    threshold: float,
) -> None:
    left, top, right, bottom = valid_crop_bounds(
        meta.out_x,
        meta.out_y,
        patch_px,
        out_w,
        out_h,
        halo_px,
    )
    gx1 = meta.out_x + left
    gy1 = meta.out_y + top
    gx2 = min(meta.out_x + right, out_w)
    gy2 = min(meta.out_y + bottom, out_h)

    if gx2 <= gx1 or gy2 <= gy1:
        return

    crop = probs[top : top + (gy2 - gy1), left : left + (gx2 - gx1)]
    binary = (crop >= threshold).astype(np.uint8)
    mask[gy1:gy2, gx1:gx2] = binary


def run_baseline(cfg: AppConfig) -> RunSummary:
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.runtime.device)
    if cfg.runtime.torch_num_threads > 0:
        torch.set_num_threads(cfg.runtime.torch_num_threads)

    start_time = time.perf_counter()

    with OpenSlideReader(cfg.paths.slide_path) as slide:
        slide.set_cache(cfg.runtime.openslide_cache_bytes)
        selection = slide.choose_level(cfg.model.target_mpp)
        out_w, out_h = slide.output_shape(cfg.model.target_mpp)

        mask_memmap_path = cfg.paths.output_dir / "mask.tmp.npy"
        mask = create_mask_memmap(mask_memmap_path, shape=(out_h, out_w))

        model = load_torchscript_model(cfg.paths.model_path, device)

        xs = axis_positions(out_w, cfg.model.patch_px, cfg.model.stride_px)
        ys = axis_positions(out_h, cfg.model.patch_px, cfg.model.stride_px)

        patches: list[np.ndarray] = []
        metas: list[PatchMeta] = []
        num_patches = 0

        for out_y in ys:
            for out_x in xs:
                patch = _read_patch(
                    slide,
                    selection,
                    out_x=out_x,
                    out_y=out_y,
                    target_mpp=cfg.model.target_mpp,
                    patch_px=cfg.model.patch_px,
                )
                patches.append(patch)
                metas.append(PatchMeta(out_x=out_x, out_y=out_y))
                num_patches += 1

                if len(patches) >= cfg.model.batch_size:
                    _infer_and_write_batch(mask, model, patches, metas, cfg, device, out_w, out_h)
                    patches.clear()
                    metas.clear()

        if patches:
            _infer_and_write_batch(mask, model, patches, metas, cfg, device, out_w, out_h)

        mask.flush()

        mask_tiff_path = None
        if cfg.output.write_tiff:
            mask_tiff_path = export_mask_tiff(
                mask,
                cfg.paths.output_dir / "mask.tif",
                bigtiff=cfg.output.bigtiff,
                compression=cfg.output.compression,
            )

        previews = save_previews(
            slide, mask, cfg.paths.output_dir, max_size=cfg.output.preview_max_size
        )

        elapsed = time.perf_counter() - start_time
        run_json_path = cfg.paths.output_dir / "run.json"
        dump_json(
            {
                "slide": asdict(slide.metadata),
                "selection": asdict(selection),
                "output_shape": {"width": out_w, "height": out_h},
                "num_patches": num_patches,
                "device": str(device),
                "timing": {"total_seconds": elapsed},
                "config": cfg.model_dump(),
                "coordinate_mapping": {
                    "level0_x_from_mask_x": "x_l0 = x_mask * target_mpp / mpp_x",
                    "level0_y_from_mask_y": "y_l0 = y_mask * target_mpp / mpp_y",
                },
            },
            run_json_path,
        )

    if not cfg.output.keep_memmap and mask_memmap_path.exists():
        mask_memmap_path.unlink()

    artifacts = RunArtifacts(
        mask_memmap=mask_memmap_path,
        mask_tiff=mask_tiff_path,
        preview_overlay=previews.get("preview_overlay"),
        preview_mask=previews.get("preview_mask"),
        run_json=run_json_path,
    )
    return RunSummary(
        slide_path=cfg.paths.slide_path,
        output_shape=(out_h, out_w),
        num_patches=num_patches,
        device=str(device),
        total_seconds=elapsed,
        artifacts=artifacts,
    )


def _infer_and_write_batch(
    mask: np.ndarray,
    model: torch.jit.ScriptModule,
    patches: list[np.ndarray],
    metas: list[PatchMeta],
    cfg: AppConfig,
    device: torch.device,
    out_w: int,
    out_h: int,
) -> None:
    batch = batch_to_tensor(patches, device)
    use_amp = cfg.runtime.use_amp and supports_amp(device)
    with torch.inference_mode():
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            raw = model(batch)
        probs_t = output_to_probs(raw, cfg.model.patch_px, apply_sigmoid=cfg.model.apply_sigmoid)
    probs = probs_t.detach().cpu().numpy()

    for prob, meta in zip(probs, metas, strict=True):
        _write_patch(
            mask,
            prob,
            meta,
            patch_px=cfg.model.patch_px,
            halo_px=cfg.model.halo_px,
            out_w=out_w,
            out_h=out_h,
            threshold=cfg.model.threshold,
        )
