from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from wsi_seg.config import AppConfig
from wsi_seg.geometry import PatchMeta, valid_crop_bounds
from wsi_seg.logging_utils import StructuredRunLogger
from wsi_seg.model import batch_to_tensor, load_torchscript_model, output_to_probs
from wsi_seg.prefetch import ReaderMetrics, SuperTilePrefetcher
from wsi_seg.preview import save_previews
from wsi_seg.scheduler import (
    PlanningSummary,
    SuperTilePlan,
    group_patches_into_supertiles,
    plan_patch_grid,
    schedule_roi,
)
from wsi_seg.slide import LevelSelection, OpenSlideReader, read_output_region
from wsi_seg.tissue import build_coarse_tissue_mask
from wsi_seg.utils import (
    dump_json,
    generate_run_id,
    git_info,
    resolve_device,
    supports_amp,
    utc_now_iso,
)
from wsi_seg.writer import create_mask_memmap, export_mask_ome_tiff, export_mask_tiff


@dataclass(slots=True)
class WallTiming:
    open_slide: float = 0.0
    plan_and_mask: float = 0.0
    load_model: float = 0.0
    processing_loop: float = 0.0
    export_outputs: float = 0.0
    total: float = 0.0


@dataclass(slots=True)
class ComponentTiming:
    reader_active: float = 0.0
    reader_wait: float = 0.0
    model_infer: float = 0.0
    writeback: float = 0.0


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
    mask_ome_tiff: Path | None
    preview_overlay: Path | None
    preview_mask: Path | None
    preview_tissue: Path | None
    events_jsonl: Path
    run_json: Path


def exports_requested(cfg: AppConfig) -> bool:
    return any((cfg.output.write_tiff, cfg.output.write_ome_tiff, cfg.output.write_previews))


def needs_materialized_export_mask(cfg: AppConfig) -> bool:
    return any((cfg.output.write_tiff, cfg.output.write_ome_tiff))

@dataclass(slots=True)
class RunSummary:
    run_id: str
    slide_path: Path
    output_shape: tuple[int, int]
    device: str
    wall_timing: WallTiming
    component_timing: ComponentTiming
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



def run_baseline(cfg: AppConfig, *, verbose: bool = False) -> RunSummary:
    run_id, run_dir = _resolve_run_dir(cfg)
    run_logger = StructuredRunLogger(run_dir, verbose=verbose)
    started_at = utc_now_iso()
    device = resolve_device(cfg.runtime.device)
    if cfg.runtime.torch_num_threads > 0:
        torch.set_num_threads(cfg.runtime.torch_num_threads)

    wall = WallTiming()
    components = ComponentTiming()
    batch_stats = BatchStats()
    wall_start = time.perf_counter()

    t0 = time.perf_counter()
    slide = OpenSlideReader(
        cfg.paths.slide_path,
        mpp_override_x=cfg.slide.mpp_override_x,
        mpp_override_y=cfg.slide.mpp_override_y,
    )
    cache_enabled = slide.set_cache(cfg.runtime.openslide_cache_bytes)
    selection = slide.choose_level(
        cfg.model.target_mpp,
        policy=cfg.model.level_selection_policy,
        max_native_oversample_factor=cfg.model.max_native_oversample_factor,
    )
    wall.open_slide = time.perf_counter() - t0
    run_logger.event(
        "slide_opened",
        slide_path=cfg.paths.slide_path,
        vendor=slide.metadata.vendor,
        backend=slide.metadata.backend,
        detected_format=slide.metadata.detected_format,
        cache_enabled=cache_enabled,
        mpp_source=slide.metadata.mpp_source,
        chosen_level=selection.level,
        chosen_level_policy=selection.policy,
        chosen_level_mpp_x=selection.level_mpp_x,
        chosen_level_mpp_y=selection.level_mpp_y,
        resize_factor_x=selection.resize_factor_x,
        resize_factor_y=selection.resize_factor_y,
    )

    mask_tiff_path: Path | None = None
    mask_ome_tiff_path: Path | None = None
    previews: dict[str, Path] = {}
    mask_memmap_path = run_dir / "mask.tmp.npy"
    run_json_path = run_dir / "run.json"

    try:
        t0 = time.perf_counter()
        out_w, out_h, planning, supertiles, coarse_mask = plan_run(cfg, slide)
        wall.plan_and_mask = time.perf_counter() - t0
        run_logger.event(
            "planning_complete",
            output_width=out_w,
            output_height=out_h,
            roi=planning.roi,
            total_grid_patches=planning.total_grid_patches,
            roi_patches=planning.roi_patches,
            candidate_patches=planning.tissue_patches,
            supertiles=planning.supertiles,
        )

        t0 = time.perf_counter()
        mask = create_mask_memmap(mask_memmap_path, shape=(out_h, out_w))
        model = load_torchscript_model(cfg.paths.model_path, device)
        wall.load_model = time.perf_counter() - t0
        run_logger.event(
            "model_loaded",
            model_path=cfg.paths.model_path,
            device=str(device),
            batch_size=cfg.model.batch_size,
            use_amp=cfg.runtime.use_amp,
        )

        loop_t0 = time.perf_counter()
        use_prefetch = cfg.runtime.prefetch_supertiles and len(supertiles) > 1
        run_logger.event(
            "processing_started",
            prefetch_enabled=use_prefetch,
            prefetch_queue_size=cfg.runtime.prefetch_queue_size,
            total_supertiles=len(supertiles),
        )
        if use_prefetch:
            with SuperTilePrefetcher(
                slide_path=cfg.paths.slide_path,
                selection=selection,
                target_mpp=cfg.model.target_mpp,
                plans=supertiles,
                openslide_cache_bytes=cfg.runtime.openslide_cache_bytes,
                queue_size=cfg.runtime.prefetch_queue_size,
                mpp_override_x=cfg.slide.mpp_override_x,
                mpp_override_y=cfg.slide.mpp_override_y,
            ) as prefetcher:
                _process_prefetched_supertiles(
                    mask,
                    model,
                    prefetcher,
                    cfg,
                    device,
                    out_w,
                    out_h,
                    planning.supertiles,
                    components,
                    batch_stats,
                    run_logger,
                )
                components.reader_active = prefetcher.metrics.active_seconds
                components.reader_wait = prefetcher.metrics.wait_seconds
        else:
            serial_metrics = ReaderMetrics()
            _process_serial_supertiles(
                mask,
                model,
                slide,
                selection,
                supertiles,
                cfg,
                device,
                out_w,
                out_h,
                serial_metrics,
                components,
                batch_stats,
                run_logger,
            )
            components.reader_active = serial_metrics.active_seconds
            components.reader_wait = serial_metrics.wait_seconds
        wall.processing_loop = time.perf_counter() - loop_t0

        mask.flush()

        if exports_requested(cfg):
            t0 = time.perf_counter()
            actual_output_mpp_x = (slide.metadata.width * slide.metadata.mpp_x) / max(out_w, 1)
            actual_output_mpp_y = (slide.metadata.height * slide.metadata.mpp_y) / max(out_h, 1)
            if needs_materialized_export_mask(cfg):
                mask *= np.uint8(255)
                mask.flush()
            if cfg.output.write_tiff:
                mask_tiff_path = export_mask_tiff(
                    mask,
                    run_dir / "mask.tif",
                    mpp_x=actual_output_mpp_x,
                    mpp_y=actual_output_mpp_y,
                    bigtiff=cfg.output.bigtiff,
                    compression=cfg.output.compression,
                    tile_size=cfg.output.tiff_tile_size,
                    description={
                        "run_id": run_id,
                        "slide": str(cfg.paths.slide_path),
                        "target_mpp": cfg.model.target_mpp,
                        "actual_output_mpp_x": actual_output_mpp_x,
                        "actual_output_mpp_y": actual_output_mpp_y,
                        "roi": asdict(planning.roi),
                    },
                )
            if cfg.output.write_ome_tiff:
                mask_ome_tiff_path = export_mask_ome_tiff(
                    mask,
                    run_dir / "mask.ome.tif",
                    mpp_x=actual_output_mpp_x,
                    mpp_y=actual_output_mpp_y,
                    bigtiff=cfg.output.bigtiff,
                    compression=cfg.output.compression,
                    tile_size=cfg.output.ome_tile_size,
                    pyramid_min_size=cfg.output.ome_pyramid_min_size,
                )
            if cfg.output.write_previews:
                previews = save_previews(
                    slide,
                    mask,
                    run_dir,
                    max_size=cfg.output.preview_max_size,
                    tissue_mask=coarse_mask,
                )
            wall.export_outputs = time.perf_counter() - t0
            run_logger.event(
                "exports_complete",
                mask_tiff=mask_tiff_path,
                mask_ome_tiff=mask_ome_tiff_path,
                previews=previews,
            )
        else:
            run_logger.event("exports_skipped", reason="all export outputs disabled")

        wall.total = time.perf_counter() - wall_start
        finished_at = utc_now_iso()

        gp = planning.total_grid_patches
        rp = planning.roi_patches
        cp = planning.tissue_patches
        safe_elapsed = max(wall.total, 1e-9)
        processing_elapsed = max(wall.processing_loop, 1e-9)

        dump_json(
            {
                "run_id": run_id,
                "started_at_utc": started_at,
                "finished_at_utc": finished_at,
                "git": git_info(),
                "slide": asdict(slide.metadata),
                "selection": asdict(selection),
                "output_shape": {"width": out_w, "height": out_h},
                "planning": {
                    "roi": asdict(planning.roi),
                    "total_grid_patches": gp,
                    "roi_patches": rp,
                    "candidate_patches": cp,
                    "supertiles": planning.supertiles,
                    "edge_patches": planning.edge_patches,
                    "padded_patches": planning.padded_patches,
                    "roi_area_fraction": rp / max(gp, 1),
                    "candidate_fraction_of_roi": cp / max(rp, 1),
                    "candidate_fraction_of_grid": cp / max(gp, 1),
                    "avg_candidates_per_supertile": cp / max(planning.supertiles, 1),
                    "num_batches": batch_stats.num_batches,
                    "mean_batch_fill": batch_stats.mean_batch_fill,
                },
                "device": str(device),
                "cache_enabled": cache_enabled,
                "prefetch": {
                    "enabled": use_prefetch,
                    "queue_size": cfg.runtime.prefetch_queue_size,
                },
                "timing": {
                    "wall": asdict(wall),
                    "components": asdict(components),
                    "semantics": {
                        "reader_active": (
                            "actual time spent reading/decoding/resizing supertiles; "
                            "can overlap model inference when prefetch is enabled"
                        ),
                        "reader_wait": (
                            "critical-path stall time where the main thread blocked "
                            "waiting for the next supertile"
                        ),
                        "model_infer": "actual model execution time on inference batches",
                        "writeback": (
                            "actual time spent thresholding and stitching predictions "
                            "into the output mask"
                        ),
                        "wall": (
                            "mutually exclusive wall-clock buckets; "
                            "these add to total wall time"
                        ),
                    },
                },
                "throughput": {
                    "grid_patches_per_second": gp / safe_elapsed,
                    "roi_patches_per_second": rp / safe_elapsed,
                    "candidate_patches_per_second": cp / safe_elapsed,
                    "candidate_patches_per_processing_second": cp / processing_elapsed,
                },
                "config": cfg.model_dump(),
                "coordinate_mapping": {
                    "level0_x_from_mask_x": "x_l0 = x_mask * target_mpp / mpp_x",
                    "level0_y_from_mask_y": "y_l0 = y_mask * target_mpp / mpp_y",
                },
            },
            run_json_path,
        )
        run_logger.event(
            "run_complete",
            total_wall_seconds=wall.total,
            processing_wall_seconds=wall.processing_loop,
            reader_active_seconds=components.reader_active,
            reader_wait_seconds=components.reader_wait,
            infer_active_seconds=components.model_infer,
            writeback_seconds=components.writeback,
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
        mask_ome_tiff=mask_ome_tiff_path,
        preview_overlay=previews.get("preview_overlay"),
        preview_mask=previews.get("preview_mask"),
        preview_tissue=previews.get("preview_tissue"),
        events_jsonl=run_logger.path,
        run_json=run_json_path,
    )
    return RunSummary(
        run_id=run_id,
        slide_path=cfg.paths.slide_path,
        output_shape=(out_h, out_w),
        device=str(device),
        wall_timing=wall,
        component_timing=components,
        num_grid_patches=planning.total_grid_patches,
        num_roi_patches=planning.roi_patches,
        num_candidate_patches=planning.tissue_patches,
        num_supertiles=planning.supertiles,
        patches_per_second=planning.tissue_patches / max(wall.total, 1e-9),
        artifacts=artifacts,
    )


def _process_prefetched_supertiles(
    mask: np.ndarray,
    model: torch.jit.ScriptModule,
    prefetcher: SuperTilePrefetcher,
    cfg: AppConfig,
    device: torch.device,
    out_w: int,
    out_h: int,
    total_supertiles: int,
    components: ComponentTiming,
    batch_stats: BatchStats,
    run_logger: StructuredRunLogger,
) -> None:
    for idx, item in enumerate(prefetcher, start=1):
        _infer_and_write_supertile(
            mask,
            model,
            item.image,
            item.plan,
            cfg,
            device,
            out_w,
            out_h,
            components,
            batch_stats,
        )
        if idx == 1 or idx == total_supertiles or idx % cfg.runtime.log_every_supertiles == 0:
            run_logger.event("supertile_progress", completed=idx, total=total_supertiles)


def _process_serial_supertiles(
    mask: np.ndarray,
    model: torch.jit.ScriptModule,
    slide: OpenSlideReader,
    selection: LevelSelection,
    supertiles: list[SuperTilePlan],
    cfg: AppConfig,
    device: torch.device,
    out_w: int,
    out_h: int,
    reader_metrics: ReaderMetrics,
    components: ComponentTiming,
    batch_stats: BatchStats,
    run_logger: StructuredRunLogger,
) -> None:
    total_supertiles = len(supertiles)
    for idx, plan in enumerate(supertiles, start=1):
        t0 = time.perf_counter()
        super_arr = read_output_region(
            slide,
            selection,
            out_x=plan.out_x,
            out_y=plan.out_y,
            out_w=plan.out_w,
            out_h=plan.out_h,
            target_mpp=cfg.model.target_mpp,
        )
        dt = time.perf_counter() - t0
        reader_metrics.active_seconds += dt
        reader_metrics.wait_seconds += dt
        reader_metrics.num_reads += 1
        _infer_and_write_supertile(
            mask,
            model,
            super_arr,
            plan,
            cfg,
            device,
            out_w,
            out_h,
            components,
            batch_stats,
        )
        if idx == 1 or idx == total_supertiles or idx % cfg.runtime.log_every_supertiles == 0:
            run_logger.event("supertile_progress", completed=idx, total=total_supertiles)


def _infer_and_write_supertile(
    mask: np.ndarray,
    model: torch.jit.ScriptModule,
    super_arr: np.ndarray,
    plan: SuperTilePlan,
    cfg: AppConfig,
    device: torch.device,
    out_w: int,
    out_h: int,
    components: ComponentTiming,
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
            patch = _resize_patch(Image.fromarray(patch), cfg.model.patch_px)
        patches.append(patch)
        metas.append(meta)
        if len(patches) >= cfg.model.batch_size:
            _infer_and_write_batch(
                mask,
                model,
                patches,
                metas,
                cfg,
                device,
                out_w,
                out_h,
                components,
                batch_stats,
            )
            patches.clear()
            metas.clear()
    if patches:
        _infer_and_write_batch(
            mask,
            model,
            patches,
            metas,
            cfg,
            device,
            out_w,
            out_h,
            components,
            batch_stats,
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
    components: ComponentTiming,
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
    components.model_infer += time.perf_counter() - t0

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
    components.writeback += time.perf_counter() - t0
