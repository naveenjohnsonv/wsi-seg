from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from wsi_seg.config import AppConfig
from wsi_seg.model import probe_model
from wsi_seg.pipeline import plan_run, run_baseline
from wsi_seg.slide import OpenSlideReader
from wsi_seg.utils import format_bytes, resolve_device

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


def _load_config(config_path: Path) -> AppConfig:
    return AppConfig.from_yaml(config_path)


@app.command("inspect")
def inspect_slide(
    config_path: Annotated[Path, typer.Argument(help="Path to YAML config.")],
) -> None:
    cfg = _load_config(config_path)
    with OpenSlideReader(cfg.paths.slide_path) as slide:
        selection = slide.choose_level(cfg.model.target_mpp)
        out_w, out_h, planning, _, coarse_mask = plan_run(cfg, slide)
        md = slide.metadata

        table = Table(title="Slide inspection")
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("Path", str(md.path))
        table.add_row("Vendor", str(md.vendor))
        table.add_row("Level-0 size", f"{md.width} x {md.height}")
        table.add_row("MPP x / y", f"{md.mpp_x:.6f} / {md.mpp_y:.6f}")
        table.add_row("Objective power", str(md.objective_power))
        table.add_row("Chosen level", str(selection.level))
        mpp = f"{selection.level_mpp_x:.6f} / {selection.level_mpp_y:.6f}"
        table.add_row("Chosen level MPP x / y", mpp)
        table.add_row("Target MPP", f"{cfg.model.target_mpp:.6f}")
        table.add_row("Output mask size", f"{out_w} x {out_h}")
        table.add_row("Estimated mask bytes", format_bytes(out_w * out_h))
        table.add_row(
            "Schedule ROI",
            f"x={planning.roi.x}, y={planning.roi.y}, "
            f"w={planning.roi.width}, h={planning.roi.height}",
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
    config_path: Annotated[Path, typer.Argument(help="Path to YAML config.")],
) -> None:
    cfg = _load_config(config_path)
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


def _print_run_summary(summary) -> None:
    table = Table(title="Run summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Slide", str(summary.slide_path))
    table.add_row("Device", summary.device)
    table.add_row("Output shape", f"{summary.output_shape[1]} x {summary.output_shape[0]}")
    table.add_row("Grid patches", str(summary.num_grid_patches))
    table.add_row("ROI patches", str(summary.num_roi_patches))
    table.add_row("Candidate patches", str(summary.num_candidate_patches))
    table.add_row("Supertiles", str(summary.num_supertiles))
    table.add_row("Candidate patches / sec", f"{summary.patches_per_second:.2f}")
    table.add_row("Total seconds", f"{summary.total_seconds:.2f}")
    table.add_row("Memmap", str(summary.artifacts.mask_memmap))
    table.add_row("Mask TIFF", str(summary.artifacts.mask_tiff))
    table.add_row("Preview mask", str(summary.artifacts.preview_mask))
    table.add_row("Preview overlay", str(summary.artifacts.preview_overlay))
    table.add_row("Preview tissue", str(summary.artifacts.preview_tissue))
    table.add_row("Run manifest", str(summary.artifacts.run_json))
    console.print(table)


@app.command("run")
def run_cmd(config_path: Annotated[Path, typer.Argument(help="Path to YAML config.")]) -> None:
    cfg = _load_config(config_path)
    summary = run_baseline(cfg)
    _print_run_summary(summary)


@app.command("benchmark")
def benchmark_cmd(
    config_path: Annotated[Path, typer.Argument(help="Path to YAML config.")],
) -> None:
    cfg = _load_config(config_path)
    summary = run_baseline(cfg)
    _print_run_summary(summary)


if __name__ == "__main__":  # pragma: no cover
    app()
