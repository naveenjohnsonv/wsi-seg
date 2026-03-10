# Architecture Notes

## Pipeline overview

```text
WSI
  -> metadata inspection
  -> output frame selection
  -> coarse tissue masking from thumbnail
  -> patch scheduling in output space
  -> supertile grouping
  -> native region reads + resize to target MPP
  -> batched TorchScript inference
  -> overlap-aware binary stitching
  -> OME-TIFF + previews + run manifest
```

## Core design choices

### 1. Export in an explicit output frame

The mask is produced in a defined output frame that stores:

- origin in level-0 pixels
- level-0 width and height
- output raster width and height
- actual exported MPP after integer rounding

This keeps the exported raster physically aligned with the original slide while avoiding a huge native-resolution mask canvas.

### 2. Make physical scale configurable

The model expects a target MPP, so the code does not assume a fixed pyramid level. Instead it:

- reads slide MPP from metadata when available
- allows `mpp_override_x` / `mpp_override_y` when metadata is missing or unreliable
- supports `nearest`, `prefer_higher`, and `prefer_higher_bounded` level-selection policies
- optionally caps oversampling with `max_native_oversample_factor`

This makes the resolution policy explicit rather than implicit.

### 3. Reduce I/O before model work

The pipeline prunes unnecessary work in coarse stages:

- use slide bounds when available
- reject obvious background with a thumbnail-derived tissue mask
- group nearby output patches into supertiles

This improves read locality and reduces decoder churn compared with many tiny random patch reads.

### 4. Keep memory bounded

The writable output mask is a disk-backed `numpy.open_memmap`. Predictions are stitched directly into that mask, so peak RAM does not scale with full slide area in the same way an in-memory probability canvas would.

### 5. Use overlap-aware stitching

Patches are inferred with overlap and only the valid center crop is written back for interior tiles. This is a practical trade-off that reduces boundary artifacts without needing weighted blending or a full float accumulator.

### 6. Read/infer overlap via prefetch

When enabled, a background thread reads and decodes supertiles into a bounded queue while the main thread runs model inference. This hides read latency behind GPU compute. The queue size is configurable to balance memory pressure against throughput.

### 7. Preserve operational visibility

Each run emits:

- `run.json` records everything needed to map mask pixels back to the original slide:
  - `output_frame.origin_x_level0`, `output_frame.origin_y_level0` - where the output frame starts in level-0 pixels
  - `output_frame.width_level0`, `output_frame.height_level0` - the native extent covered
  - `output_frame.actual_output_mpp_x`, `output_frame.actual_output_mpp_y` - the true physical resolution of the exported mask after integer rounding
- `events.jsonl` with structured progress events
- preview artifacts for quick visual inspection
- OME-TIFF output for downstream viewing tools

## Why this architecture works well

The `src/wsi_seg/` package is organized so each module owns a single concern:

- `config.py` - Pydantic-validated YAML config with geometry and runtime constraints
- `slide.py` - OpenSlide reader, metadata extraction, MPP fallback chain, level selection, output frame computation
- `tissue.py` - thumbnail-based coarse tissue mask with Otsu thresholding and morphological cleanup
- `scheduler.py` - ROI computation, patch grid planning with tissue gating, supertile grouping
- `geometry.py` - axis position generation and valid-crop-bounds math
- `model.py` - TorchScript loading, batch tensor conversion, output normalization
- `prefetch.py` - threaded supertile reader with bounded queue and timing metrics
- `writer.py` - memmap creation, TIFF and OME-TIFF pyramidal export
- `preview.py` - thumbnail overlay compositing with bounds-aware mask placement
- `pipeline.py` - orchestration, timing, manifest generation
- `cli.py` - typer CLI with inspect, probe-model, and run commands
- `utils.py` - device resolution, git info, run ID generation, slide discovery
- `logging_utils.py` - structured JSONL event logger

That separation keeps the pipeline easier to test, reason about, and extend.

## Timing interpretation

The pipeline records two timing views. **Wall timing** captures mutually exclusive phases that add to total elapsed time (open slide, plan, load model, processing loop, export). **Component timing** captures overlapping subsystem activity (reader active/wait, model inference, writeback). When prefetch is enabled, reader active time overlaps with model inference, so component times will sum to more than wall time - this is expected and indicates successful overlap.

When export is enabled, OME-TIFF compression and preview generation can dominate wall time on slower storage. Use `--no-exports` to isolate core pipeline throughput.

## Deliberate trade-offs

Chosen:

- OpenSlide for broad and reliable WSI access
- center-crop stitching instead of weighted probability fusion
- memmap instead of a more complex chunked intermediate
- single-process inference with optional read prefetch thread
- explicit output-frame export instead of forcing a full level-0-sized mask

Deferred:

- weighted blending / importance maps
- Zarr or other chunked cloud-native intermediates
- multi-slide distributed orchestration
- richer artifact / QC masking
- more advanced performance regression tooling

## Scaling path

The current design already separates the main scaling boundaries:

1. **slide access**
2. **inference**
3. **artifact export**

That makes the next steps fairly natural:

- process multiple slides in parallel at the job level
- swap the memmap intermediate for chunked storage when random downstream writes matter
- move artifacts and manifests to remote object storage
- add queue-level orchestration and worker pools
- expose operational metrics around read latency, queue wait, and export cost