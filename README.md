# WSI Segmentation Pipeline

Memory-efficient lesion-segmentation inference for whole-slide images.

This project runs a TorchScript binary segmentation model on large pathology slides while keeping memory bounded, reducing unnecessary slide reads, and exporting masks that remain spatially consistent with the source image.

## What it includes

- OpenSlide-based inference for common WSI formats such as **MRXS, SVS, TIFF, NDPI, SCN, BIF, VMS, and VMU**
- configurable physical-scale inference using a target MPP
- I/O-aware scheduling with optional bounds filtering, tissue masking, and supertile reads
- bounded-memory stitching via a disk-backed `numpy` memmap
- pyramidal OME-TIFF export plus preview artifacts
- structured run manifests and JSONL event logs
- tests, CI, and a reproducible `uv` / Docker setup

## Setup

### Local with `uv`

CPU:

```bash
uv sync --extra cpu --group dev
```

Other supported runtimes:

```bash
uv sync --extra cu128 --group dev
uv sync --extra xpu --group dev
uv sync --extra rocm --group dev
```

### Docker

```bash
docker build -t wsi-seg .
```

The checked-in Docker image is CPU-first for portability.

## Expected data layout

```text
data/
├── model_scripted.pt
└── slides/
    ├── slide_1.mrxs
    ├── slide_1/
    ├── slide_2.svs
    └── slide_3.tif
```

## Usage

Inspect a slide:

```bash
uv run wsi-seg inspect configs/default.yaml --slide-path data/slides/slide_1.mrxs
```

Probe the TorchScript model:

```bash
uv run wsi-seg probe-model configs/default.yaml --model-path data/model_scripted.pt
```

Run one slide:

```bash
uv run wsi-seg run configs/default.yaml --slide-path data/slides/slide_1.mrxs
```

Run a directory of slides:

```bash
uv run wsi-seg run configs/default.yaml --slide-path data/slides
```

Run recursively:

```bash
uv run wsi-seg run configs/default.yaml --slide-path data/slides --recursive
```

Disable exports:

```bash
uv run wsi-seg run configs/default.yaml --slide-path data/slides/slide_1.mrxs --no-exports
```

## Key configuration knobs

`configs/default.yaml` exposes the main runtime and inference choices:

```yaml
slide:
  mpp_override_x: null
  mpp_override_y: null

model:
  target_mpp: 0.88
  patch_px: 512
  stride_px: 384
  halo_px: 64
  batch_size: 8
  threshold: 0.5
  apply_sigmoid: true
  level_selection_policy: prefer_higher_bounded
  max_native_oversample_factor: 2.0

runtime:
  device: auto
  use_amp: true
  openslide_cache_bytes: 536870912
  torch_num_threads: 1
  prefetch_supertiles: true
  prefetch_queue_size: 6

schedule:
  use_bounds: true
  use_tissue_mask: true
  supertile_px: 4096

output:
  write_ome_tiff: true
  write_previews: true
  keep_memmap: false
```

## Outputs

Each run writes to:

```text
outputs/<slide_stem>/<run_id>/
```

Typical artifacts:

```text
mask.ome.tif
preview_mask.png
preview_overlay.png
preview_tissue.png
events.jsonl
run.json
```

`run.json` records slide metadata, chosen pyramid level, output frame, planning statistics, timings, throughput, and coordinate mapping back to level-0 space.

## Architectural decisions

- **Physical scale is explicit.** The pipeline reads from a selected native pyramid level, resizes into a target-MPP output frame, and exports the mask with the frame’s actual post-rounding MPP so physical alignment stays correct.
- **I/O is reduced before inference.** Optional bounds handling, thumbnail tissue masking, and supertile grouping reduce unnecessary random reads.
- **Memory stays bounded.** Patch predictions are stitched into a disk-backed `numpy` memmap instead of building a full-slide in-memory probability canvas.
- **Configuration is first-class.** Slide metadata overrides, level-selection policy, runtime backend, scheduling, and export behavior are YAML-driven.

## Scaling vision

The current structure cleanly separates slide reading, scheduling, inference, and export. That makes it straightforward to extend toward multi-slide orchestration, worker pools, object-storage-backed artifacts, and chunked intermediates such as Zarr without rewriting the core spatial logic.

## More documentation

- [Architecture notes](docs/ARCHITECTURE.md)
- [Running notes](docs/RUNNING.md)
- [Results placeholders](docs/RESULTS.md)