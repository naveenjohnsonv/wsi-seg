from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from wsi_seg.config import AppConfig
from wsi_seg.geometry import PatchMeta, valid_crop_bounds
from wsi_seg.model import batch_to_tensor, load_torchscript_model, output_to_probs
from wsi_seg.preview import save_previews
from wsi_seg.scheduler import (
    PlanningSummary,
    SuperTilePlan,
    group_patches_into_supertiles,
    plan_patch_grid,
    schedule_roi,
)
from wsi_seg.slide import LevelSelection, OpenSlideReader
from wsi_seg.tissue import build_coarse_tissue_mask
from wsi_seg.utils import (
    dump_json,
    generate_run_id,
    git_info,
    resolve_device,
    supports_amp,
    utc_now_iso,
)
from wsi_seg.writer import create_mask_memmap, export_mask_tiff


@dataclass(slots=True)
class StageTiming:
    open_slide: float = 0.0
    plan_and_mask: float = 0.0
    read_supertiles: float = 0.0
    model_infer: float = 0.0
    writeback: float = 0.0
    export_outputs: float = 0.0
    total: float = 0.0


@dataclass(slots=True)
class BatchStats:
    num_batches: int = 0
    total_items: int = 0

    @property
    def mean_batch_fill(self) -> float:
        return self.total_items / self.num_batches if self.num_batches else 0.0


@dataclass(slots=True)
class RunArtifacts:
    run_dir: Path
    mask_memmap: Path | None
    mask_tiff: Path | None
    preview_overlay: Path | None
    preview_mask: Path | None
    preview_tissue: Path | None
    run_json: Path


@dataclass(slots=True)
class RunSummary:
    run_id: str
    slide_path: Path
    output_shape: tuple[int, int]
    device: str
    stage_timing: StageTiming
    num_grid_patches: int
    num_roi_patches: int
    num_candidate_patches: int
    num_supertiles: int
    patches_per_second: float
    artifacts: RunArtifacts


def _resize_patch(pil_img: Image.Image, patch_px: int) -> np.ndarray:
    if pil_img.size != (patch_px, patch_px):
        pil_img = pil_img.resize((patch_px, patch_px), resample=Image.Resampling.BILINEAR)
    return np.asarray(pil_img, dtype=np.uint8)


def _read_output_region(
    slide: OpenSlideReader,
    selection: LevelSelection,
    *,
    out_x: int,
    out_y: int,
    out_w: int,
    out_h: int,
    target_mpp: float,
) -> np.ndarray:
    mpp_x = slide.metadata.mpp_x
    mpp_y = slide.metadata.mpp_y
    level_mpp_x = selection.level_mpp_x
    level_mpp_y = selection.level_mpp_y

    read_x = int(round(out_x * target_mpp / mpp_x))
    read_y = int(round(out_y * target_mpp / mpp_y))
    read_w = max(1, int(round(out_w * target_mpp / level_mpp_x)))
    read_h = max(1, int(round(out_h * target_mpp / level_mpp_y)))

    pil_img = slide.read_region_rgb((read_x, read_y), selection.level, (read_w, read_h))
    if pil_img.size != (out_w, out_h):
        pil_img = pil_img.resize((out_w, out_h), resample=Image.Resampling.BILINEAR)
    return np.asarray(pil_img, dtype=np.uint8)


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


def plan_run(
    cfg: AppConfig, slide: OpenSlideReader
) -> tuple[int, int, PlanningSummary, list[SuperTilePlan], np.ndarray | None]:
    out_w, out_h, roi = schedule_roi(slide, cfg.model.target_mpp, cfg.schedule.use_bounds)
    coarse_mask = None
    coarse_mask_array = None
    if cfg.schedule.use_tissue_mask:
        coarse_mask = build_coarse_tissue_mask(
            slide,
            max_size=cfg.schedule.tissue_mask_max_size,
            saturation_threshold=cfg.schedule.tissue_mask_saturation_threshold,
            white_threshold=cfg.schedule.tissue_mask_white_threshold,
        )
        coarse_mask_array = coarse_mask.mask

    patch_metas, planning = plan_patch_grid(
        out_w=out_w,
        out_h=out_h,
        patch_px=cfg.model.patch_px,
        stride_px=cfg.model.stride_px,
        halo_px=cfg.model.halo_px,
        roi=roi,
        coarse_mask=coarse_mask,
        min_tissue_fraction=cfg.schedule.tissue_mask_min_fraction,
    )
    supertiles = group_patches_into_supertiles(
        patch_metas,
        supertile_px=cfg.schedule.supertile_px,
        patch_px=cfg.model.patch_px,
    )
    planning.supertiles = len(supertiles)
    return out_w, out_h, planning, supertiles, coarse_mask_array


def _resolve_run_dir(cfg: AppConfig) -> tuple[str, Path]:
    run_id = generate_run_id(cfg.model_dump())
    slide_stem = cfg.paths.slide_path.stem
    run_dir = cfg.paths.output_dir / slide_stem / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def run_baseline(cfg: AppConfig) -> RunSummary:
    run_id, run_dir = _resolve_run_dir(cfg)
    started_at = utc_now_iso()
    device = resolve_device(cfg.runtime.device)
    if cfg.runtime.torch_num_threads > 0:
        torch.set_num_threads(cfg.runtime.torch_num_threads)

    timing = StageTiming()
    batch_stats = BatchStats()
    wall_start = time.perf_counter()

    # --- stage: open slide ---
    t0 = time.perf_counter()
    slide = OpenSlideReader(cfg.paths.slide_path)
    cache_enabled = slide.set_cache(cfg.runtime.openslide_cache_bytes)
    selection = slide.choose_level(cfg.model.target_mpp)
    timing.open_slide = time.perf_counter() - t0

    try:
        # --- stage: plan and mask ---
        t0 = time.perf_counter()
        out_w, out_h, planning, supertiles, coarse_mask = plan_run(cfg, slide)
        timing.plan_and_mask = time.perf_counter() - t0

        mask_memmap_path = run_dir / "mask.tmp.npy"
        mask = create_mask_memmap(mask_memmap_path, shape=(out_h, out_w))
        model = load_torchscript_model(cfg.paths.model_path, device)

        # --- stages: read / infer / writeback (accumulated) ---
        for plan in supertiles:
            t0 = time.perf_counter()
            super_arr = _read_output_region(
                slide,
                selection,
                out_x=plan.out_x,
                out_y=plan.out_y,
                out_w=plan.out_w,
                out_h=plan.out_h,
                target_mpp=cfg.model.target_mpp,
            )
            timing.read_supertiles += time.perf_counter() - t0

            _infer_and_write_supertile(
                mask, model, super_arr, plan, cfg, device, out_w, out_h, timing, batch_stats
            )

        mask.flush()

        # --- stage: export outputs ---
        t0 = time.perf_counter()
        mask_tiff_path = None
        if cfg.output.write_tiff:
            mask_tiff_path = export_mask_tiff(
                mask,
                run_dir / "mask.tif",
                bigtiff=cfg.output.bigtiff,
                compression=cfg.output.compression,
            )

        previews: dict[str, Path] = {}
        if cfg.output.write_previews:
            previews = save_previews(
                slide,
                mask,
                run_dir,
                max_size=cfg.output.preview_max_size,
                tissue_mask=coarse_mask,
            )
        timing.export_outputs = time.perf_counter() - t0

        timing.total = time.perf_counter() - wall_start
        finished_at = utc_now_iso()

        elapsed = timing.total
        gi = git_info()
        tp = planning.tissue_patches
        rp = planning.roi_patches
        gp = planning.total_grid_patches
        safe_elapsed = max(elapsed, 1e-9)
        st_count = planning.supertiles

        run_json_path = run_dir / "run.json"
        dump_json(
            {
                "run_id": run_id,
                "started_at_utc": started_at,
                "finished_at_utc": finished_at,
                "git": gi,
                "slide": asdict(slide.metadata),
                "selection": asdict(selection),
                "output_shape": {"width": out_w, "height": out_h},
                "planning": {
                    "roi": asdict(planning.roi),
                    "total_grid_patches": gp,
                    "roi_patches": rp,
                    "candidate_patches": tp,
                    "supertiles": st_count,
                    "edge_patches": planning.edge_patches,
                    "padded_patches": planning.padded_patches,
                    "roi_area_fraction": rp / max(gp, 1),
                    "candidate_fraction_of_roi": tp / max(rp, 1),
                    "candidate_fraction_of_grid": tp / max(gp, 1),
                    "avg_candidates_per_supertile": tp / max(st_count, 1),
                    "num_batches": batch_stats.num_batches,
                    "mean_batch_fill": batch_stats.mean_batch_fill,
                },
                "device": str(device),
                "cache_enabled": cache_enabled,
                "timing": asdict(timing),
                "throughput": {
                    "grid_patches_per_second": gp / safe_elapsed,
                    "roi_patches_per_second": rp / safe_elapsed,
                    "candidate_patches_per_second": tp / safe_elapsed,
                },
                "config": cfg.model_dump(),
                "coordinate_mapping": {
                    "level0_x_from_mask_x": "x_l0 = x_mask * target_mpp / mpp_x",
                    "level0_y_from_mask_y": "y_l0 = y_mask * target_mpp / mpp_y",
                },
            },
            run_json_path,
        )
    finally:
        slide.close()

    mask_memmap_artifact: Path | None = mask_memmap_path
    if not cfg.output.keep_memmap and mask_memmap_path.exists():
        mask_memmap_path.unlink()
        mask_memmap_artifact = None

    artifacts = RunArtifacts(
        run_dir=run_dir,
        mask_memmap=mask_memmap_artifact,
        mask_tiff=mask_tiff_path,
        preview_overlay=previews.get("preview_overlay"),
        preview_mask=previews.get("preview_mask"),
        preview_tissue=previews.get("preview_tissue"),
        run_json=run_json_path,
    )
    return RunSummary(
        run_id=run_id,
        slide_path=cfg.paths.slide_path,
        output_shape=(out_h, out_w),
        device=str(device),
        stage_timing=timing,
        num_grid_patches=planning.total_grid_patches,
        num_roi_patches=planning.roi_patches,
        num_candidate_patches=planning.tissue_patches,
        num_supertiles=planning.supertiles,
        patches_per_second=planning.tissue_patches / max(elapsed, 1e-9),
        artifacts=artifacts,
    )


def _infer_and_write_supertile(
    mask: np.ndarray,
    model: torch.jit.ScriptModule,
    super_arr: np.ndarray,
    plan: SuperTilePlan,
    cfg: AppConfig,
    device: torch.device,
    out_w: int,
    out_h: int,
    timing: StageTiming,
    batch_stats: BatchStats,
) -> None:
    patches: list[np.ndarray] = []
    metas: list[PatchMeta] = []
    for meta in plan.patches:
        local_x = meta.out_x - plan.out_x
        local_y = meta.out_y - plan.out_y
        ppx = cfg.model.patch_px
        patch = super_arr[local_y : local_y + ppx, local_x : local_x + ppx]
        if patch.shape[:2] != (cfg.model.patch_px, cfg.model.patch_px):
            pil = Image.fromarray(patch)
            patch = _resize_patch(pil, cfg.model.patch_px)
        patches.append(patch)
        metas.append(meta)
        if len(patches) >= cfg.model.batch_size:
            _infer_and_write_batch(
                mask, model, patches, metas, cfg, device, out_w, out_h, timing, batch_stats
            )
            patches.clear()
            metas.clear()
    if patches:
        _infer_and_write_batch(
            mask, model, patches, metas, cfg, device, out_w, out_h, timing, batch_stats
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
    timing: StageTiming,
    batch_stats: BatchStats,
) -> None:
    batch_stats.num_batches += 1
    batch_stats.total_items += len(patches)

    batch = batch_to_tensor(patches, device)
    use_amp = cfg.runtime.use_amp and supports_amp(device)

    t0 = time.perf_counter()
    with torch.inference_mode():
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            raw = model(batch)
        probs_t = output_to_probs(raw, cfg.model.patch_px, apply_sigmoid=cfg.model.apply_sigmoid)
    probs = probs_t.detach().cpu().numpy()
    timing.model_infer += time.perf_counter() - t0

    t0 = time.perf_counter()
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
    timing.writeback += time.perf_counter() - t0
