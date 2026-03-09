from __future__ import annotations

import csv
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from wsi_seg.config import AppConfig
from wsi_seg.logging_utils import configure_logging
from wsi_seg.model import probe_model
from wsi_seg.pipeline import RunSummary, plan_run, run_baseline
from wsi_seg.slide import OpenSlideReader
from wsi_seg.utils import (
    SUPPORTED_SLIDE_SUFFIXES,
    discover_slide_paths,
    dump_json,
    format_bytes,
    generate_run_id,
    resolve_device,
)

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()
_DEFAULT_SLIDES_DIR = Path("data/slides")



def _load_config(config_path: Path) -> AppConfig:
    return AppConfig.from_yaml(config_path)



def _cfg_with_slide(cfg: AppConfig, slide_path: Path) -> AppConfig:
    out = cfg.model_copy(deep=True)
    out.paths.slide_path = slide_path.expanduser().resolve()
    return out



def _resolve_slides(
    cfg: AppConfig,
    slide_path_override: Path | None,
    *,
    recursive: bool,
) -> tuple[Path, list[Path]]:
    if slide_path_override is not None:
        target = slide_path_override.expanduser().resolve()
    elif cfg.paths.slide_path is not None:
        target = cfg.paths.slide_path
    else:
        target = _DEFAULT_SLIDES_DIR.resolve()
    return target, discover_slide_paths(target, recursive=recursive)



def _supported_suffixes_text() -> str:
    return ", ".join(SUPPORTED_SLIDE_SUFFIXES)


@app.command("inspect")
def inspect_slide(
    config_path: Annotated[Path, typer.Argument(help="Path to YAML config.")] = Path("configs/default.yaml"),
    slide_path: Annotated[
        Path | None,
        typer.Option("--slide-path", help="Slide file or directory to inspect."),
    ] = None,
    recursive: Annotated[bool, typer.Option("--recursive", help="Recurse into directories.")] = False,
) -> None:
    cfg = _load_config(config_path)
    target, slides = _resolve_slides(cfg, slide_path, recursive=recursive)
    if not slides:
        raise typer.BadParameter(
            f"No supported slide files found in {target}. Supported suffixes: {_supported_suffixes_text()}"
        )

    for sp in slides:
        slide_cfg = _cfg_with_slide(cfg, sp)
        with OpenSlideReader(slide_cfg.paths.slide_path) as slide:
            selection = slide.choose_level(slide_cfg.model.target_mpp)
            out_w, out_h, planning, _, coarse_mask = plan_run(slide_cfg, slide)
            md = slide.metadata

            table = Table(title=f"Slide inspection — {sp.name}")
            table.add_column("Field")
            table.add_column("Value")
            table.add_row("Path", str(md.path))
            table.add_row("Vendor", str(md.vendor))
            table.add_row("Level-0 size", f"{md.width} x {md.height}")
            table.add_row("MPP x / y", f"{md.mpp_x:.6f} / {md.mpp_y:.6f}")
            table.add_row("MPP source", md.mpp_source)
            table.add_row("Objective power", str(md.objective_power))
            table.add_row("Chosen level", str(selection.level))
            table.add_row(
                "Chosen level MPP x / y",
                f"{selection.level_mpp_x:.6f} / {selection.level_mpp_y:.6f}",
            )
            table.add_row("Target MPP", f"{slide_cfg.model.target_mpp:.6f}")
            table.add_row("Output mask size", f"{out_w} x {out_h}")
            table.add_row("Estimated mask bytes", format_bytes(out_w * out_h))
            table.add_row(
                "Schedule ROI",
                f"x={planning.roi.x}, y={planning.roi.y}, w={planning.roi.width}, h={planning.roi.height}",
            )
            table.add_row("Grid patches", str(planning.total_grid_patches))
            table.add_row("ROI patches", str(planning.roi_patches))
            table.add_row("Candidate patches", str(planning.tissue_patches))
            table.add_row("Supertiles", str(planning.supertiles))
            if coarse_mask is not None:
                table.add_row("Coarse tissue mask", f"{coarse_mask.shape[1]} x {coarse_mask.shape[0]}")
            if md.bounds is not None:
                table.add_row(
                    "Bounds",
                    f"x={md.bounds.x}, y={md.bounds.y}, w={md.bounds.width}, h={md.bounds.height}",
                )
            console.print(table)

            levels = Table(title="Pyramid levels")
            levels.add_column("Level")
            levels.add_column("Dimensions")
            levels.add_column("Downsample")
            levels.add_column("MPP x / y")
            for level in md.levels:
                levels.add_row(
                    str(level.index),
                    f"{level.width} x {level.height}",
                    f"{level.downsample:.4f}",
                    f"{level.mpp_x:.6f} / {level.mpp_y:.6f}",
                )
            console.print(levels)


@app.command("probe-model")
def probe_model_cmd(
    config_path: Annotated[Path, typer.Argument(help="Path to YAML config.")] = Path("configs/default.yaml"),
    model_path: Annotated[
        Path | None,
        typer.Option("--model-path", help="Override model path from config."),
    ] = None,
) -> None:
    cfg = _load_config(config_path)
    if model_path is not None:
        cfg = cfg.model_copy(deep=True)
        cfg.paths.model_path = model_path.expanduser().resolve()
    device = resolve_device(cfg.runtime.device)
    probe = probe_model(cfg.paths.model_path, device, cfg.model.patch_px)

    table = Table(title="TorchScript model probe")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Model path", str(probe.model_path))
    table.add_row("Device", probe.device)
    table.add_row("Input shape", str(probe.input_shape))
    table.add_row("Output shape", str(probe.output_shape))
    table.add_row("Output kind", probe.output_kind)
    console.print(table)



def _print_run_summary(summary: RunSummary) -> None:
    s = summary
    table = Table(title="Run summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Run ID", s.run_id)
    table.add_row("Run directory", str(s.artifacts.run_dir))
    table.add_row("Slide", str(s.slide_path))
    table.add_row("Device", s.device)
    table.add_row("Output shape", f"{s.output_shape[1]} x {s.output_shape[0]}")
    table.add_row("Grid patches", str(s.num_grid_patches))
    table.add_row("ROI patches", str(s.num_roi_patches))
    table.add_row("Candidate patches", str(s.num_candidate_patches))
    table.add_row("Supertiles", str(s.num_supertiles))
    table.add_row("Candidate patches / sec", f"{s.patches_per_second:.2f}")
    console.print(table)

    wall = s.wall_timing
    wall_table = Table(title="Wall timing (seconds)")
    wall_table.add_column("Bucket")
    wall_table.add_column("Seconds", justify="right")
    wall_table.add_column("% of total", justify="right")
    total = max(wall.total, 1e-9)
    for label, value in [
        ("Open slide", wall.open_slide),
        ("Plan + tissue mask", wall.plan_and_mask),
        ("Load model", wall.load_model),
        ("Processing loop", wall.processing_loop),
        ("Export outputs", wall.export_outputs),
    ]:
        wall_table.add_row(label, f"{value:.3f}", f"{100 * value / total:.1f}%")
    wall_table.add_row("Total", f"{wall.total:.3f}", "100.0%")
    console.print(wall_table)

    comp = s.component_timing
    comp_table = Table(title="Component timing (clean overlap semantics)")
    comp_table.add_column("Metric")
    comp_table.add_column("Seconds", justify="right")
    comp_table.add_row("Reader active", f"{comp.reader_active:.3f}")
    comp_table.add_row("Reader wait", f"{comp.reader_wait:.3f}")
    comp_table.add_row("Model infer active", f"{comp.model_infer:.3f}")
    comp_table.add_row("Writeback active", f"{comp.writeback:.3f}")
    console.print(comp_table)

    artifacts_table = Table(title="Artifacts")
    artifacts_table.add_column("File")
    artifacts_table.add_column("Path")
    for label, path in [
        ("Memmap", s.artifacts.mask_memmap),
        ("Mask TIFF", s.artifacts.mask_tiff),
        ("Mask OME-TIFF", s.artifacts.mask_ome_tiff),
        ("Preview mask", s.artifacts.preview_mask),
        ("Preview overlay", s.artifacts.preview_overlay),
        ("Preview tissue", s.artifacts.preview_tissue),
        ("Events JSONL", s.artifacts.events_jsonl),
        ("Run manifest", s.artifacts.run_json),
    ]:
        if path is not None:
            artifacts_table.add_row(label, str(path))
    console.print(artifacts_table)



def _print_batch_summary(summaries: list[RunSummary], cfg: AppConfig) -> None:
    batch_rows = [
        {
            "slide_stem": s.slide_path.stem,
            "run_id": s.run_id,
            "device": s.device,
            "grid_patches": s.num_grid_patches,
            "roi_patches": s.num_roi_patches,
            "candidate_patches": s.num_candidate_patches,
            "supertiles": s.num_supertiles,
            "seconds": round(s.wall_timing.total, 6),
            "candidate_patches_per_second": round(s.patches_per_second, 6),
            "run_dir": str(s.artifacts.run_dir),
            "run_json": str(s.artifacts.run_json),
        }
        for s in summaries
    ]
    batch_id = f"batch_{generate_run_id(cfg.model_dump())}"
    batch_dir = cfg.paths.output_dir / "_batches" / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    csv_path = batch_dir / "summary.csv"
    json_path = batch_dir / "summary.json"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(batch_rows[0].keys()))
        writer.writeheader()
        writer.writerows(batch_rows)
    dump_json({"batch_id": batch_id, "num_slides": len(batch_rows), "slides": batch_rows}, json_path)

    table = Table(title="Batch run summary")
    table.add_column("Slide")
    table.add_column("Run ID")
    table.add_column("Seconds", justify="right")
    table.add_column("Candidates", justify="right")
    table.add_column("Supertiles", justify="right")
    for row in batch_rows:
        table.add_row(
            row["slide_stem"],
            row["run_id"],
            f'{row["seconds"]:.2f}',
            str(row["candidate_patches"]),
            str(row["supertiles"]),
        )
    console.print(table)
    console.print(f"Batch directory: {batch_dir}")


@app.command("run")
def run_cmd(
    config_path: Annotated[Path, typer.Argument(help="Path to YAML config.")] = Path("configs/default.yaml"),
    slide_path: Annotated[
        Path | None,
        typer.Option("--slide-path", help="Slide file or directory. Default: data/slides."),
    ] = None,
    model_path: Annotated[
        Path | None,
        typer.Option("--model-path", help="Override model path from config."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Override output root from config."),
    ] = None,
    no_exports: Annotated[
        bool,
        typer.Option(
            "--no-exports/--with-exports",
            help="Skip TIFF and preview export to isolate core pipeline timing.",
        ),
    ] = False,
    keep_memmap: Annotated[
        bool | None,
        typer.Option(
            "--keep-memmap/--discard-memmap",
            help="Keep or discard the intermediate memmap after the run.",
        ),
    ] = None,
    recursive: Annotated[bool, typer.Option("--recursive", help="Recurse into directories.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", help="Enable richer logging.")] = False,
) -> None:
    configure_logging(verbose=verbose)
    cfg = _load_config(config_path)
    if model_path is not None:
        cfg = cfg.model_copy(deep=True)
        cfg.paths.model_path = model_path.expanduser().resolve()
    if output_dir is not None:
        cfg = cfg.model_copy(deep=True)
        cfg.paths.output_dir = output_dir.expanduser().resolve()
    if no_exports:
        cfg = cfg.model_copy(deep=True)
        cfg.output.write_tiff = False
        cfg.output.write_ome_tiff = False
        cfg.output.write_previews = False
    if keep_memmap is not None:
        cfg = cfg.model_copy(deep=True)
        cfg.output.keep_memmap = keep_memmap

    target, slides = _resolve_slides(cfg, slide_path, recursive=recursive)
    if not slides:
        raise typer.BadParameter(
            f"No supported slide files found in {target}. Supported suffixes: {_supported_suffixes_text()}"
        )

    summaries: list[RunSummary] = []
    for sp in slides:
        slide_cfg = _cfg_with_slide(cfg, sp)
        summary = run_baseline(slide_cfg, verbose=verbose)
        summaries.append(summary)
        _print_run_summary(summary)

    if len(summaries) > 1:
        _print_batch_summary(summaries, cfg)


if __name__ == "__main__":  # pragma: no cover
    app()
