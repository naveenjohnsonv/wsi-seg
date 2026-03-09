# WSI Segmentation Pipeline

A production-style whole-slide image (WSI) lesion-segmentation inference pipeline built on **OpenSlide** and **PyTorch**.

It is designed around the constraints of digital pathology:

- **massive** multi-resolution whole-slide images
- model inputs defined in **physical resolution** (`target_mpp`), not just nominal magnification
- **bounded memory** via disk-backed intermediate outputs
- **I/O-aware scheduling** through bounds-aware planning, conservative coarse tissue masking, and supertile reads

## Current implemented phases

### Phase 0 — project scaffold
- CLI (`inspect`, `probe-model`, `run`)
- YAML config with validation
- Docker / `uv` / CI / tests

### Phase 1 — correctness baseline
- OpenSlide-based MIRAX/MRXS reader
- robust metadata inspection
- target-MPP-aware level selection
- TorchScript model probing
- overlap + center-crop stitching
- memmap-backed mask writing
- TIFF export + preview overlay + `run.json`

### Phase 2 — first performance pass
- **bounds-aware scheduling** in output space
- **coarse tissue mask** built from a thumbnail to skip obvious background
- **supertile-based reading** instead of per-patch `read_region()`
- richer planning and throughput stats in `inspect` and `run`

### Phase 3 — run history and observability
- **per-run output directories**: `outputs/<slide_stem>/<run_id>/`
- **run ID** format: `<UTC timestamp>_<git-sha7>_<config-hash8>`
- **wall + component timing**: open slide, plan+mask, load model, processing loop, export; reader active/wait, model infer, writeback
- **three throughput tiers**: grid/ROI/candidate patches per second
- **planning fractions**: ROI area fraction, candidate fraction of ROI and grid
- **batch stats**: number of batches, mean batch fill
- **git traceability**: commit SHA, branch, dirty flag in `run.json`

### Phase 4 — unified CLI
- **single `run` command** replaces `run`, `run-many`, and `benchmark`
- `--slide-path` accepts a file (single slide) or directory (batch); defaults to `data/slides`
- `--recursive` recurses into directories for slide discovery
- `--verbose` enables structured event logging
- `--no-exports` isolates core pipeline timing without TIFF/preview cost
- **intermediate memmaps are discarded by default**; `--keep-memmap` preserves them for debugging
- **no paths in `default.yaml`** — sensible defaults live in code

### Phase 5 — async prefetch, OME-TIFF pyramid, structured logging
- **async supertile prefetch**: background reader thread with queue-based pipeline overlapping I/O and inference
- **pyramidal OME-TIFF export** (`mask.ome.tif`): SubIFD pyramid levels, tiled BigTIFF, zlib compression, OME `PhysicalSizeX`/`PhysicalSizeY` metadata — QuPath-compatible
- **OME-TIFF is the default output**; flat TIFF (`mask.tif`) disabled by default, kept as an optional processing artifact
- **mask stored as 0/255** (not 0/1) for standalone viewing usability
- **lazy pyramid generation**: levels are computed and written one at a time, not materialized as a list, keeping peak memory bounded
- **in-place memmap scaling**: 0→255 scaling happens on the memmap without allocating a full copy
- **structured run logger**: `events.jsonl` with timestamped events throughout the pipeline
- **multi-source MPP extraction**: openslide properties, Aperio keys, text description, TIFF resolution tags, objective power estimation
- **multi-format slide discovery**: `.mrxs`, `.svs`, `.ndpi`, `.tif`, `.tiff`, `.scn`, `.bif`, `.vms`, `.vmu`

## Chosen architecture

- backend: **OpenSlide**
- scale policy: **target MPP is authoritative**
- read level: **nearest level by MPP error, then resize once**
- schedule ROI: **slide bounds projected to output space**
- pruning: **conservative coarse tissue mask**
- read granularity: **supertile reads**
- stitching: **overlap + center-crop writeback**
- intermediate storage: **NumPy memmap**
- final outputs: **pyramidal OME-TIFF + previews + run manifest**

## Repo layout

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/
│   └── default.yaml
├── src/
│   └── wsi_seg/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── geometry.py
│       ├── logging_utils.py
│       ├── model.py
│       ├── pipeline.py
│       ├── prefetch.py
│       ├── preview.py
│       ├── scheduler.py
│       ├── slide.py
│       ├── tissue.py
│       ├── utils.py
│       └── writer.py
├── tests/
│   ├── test_config.py
│   ├── test_pipeline_math.py
│   ├── test_scheduler.py
│   ├── test_slide_metadata.py
│   ├── test_utils.py
│   └── test_writer.py
├── .gitignore
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then sync with the extra that matches your device:

```bash
uv sync --extra xpu   --group dev   # Intel Arc / Xe
uv sync --extra cu128 --group dev   # NVIDIA CUDA 12.8
uv sync --extra rocm  --group dev   # AMD ROCm 6.4
uv sync --extra cpu   --group dev   # CPU only / CI
```

Docker builds install the **CPU** extra by default so the image is reproducible without GPU-specific wheels.

## Data layout

```text
data/
├── model_scripted.pt
└── slides/
    ├── MJUL22785295_001.mrxs
    └── MJUL22785295_001/
```

## Commands

Run all slides in `data/slides/` with defaults:

```bash
uv run wsi-seg run
```

Run a single slide:

```bash
uv run wsi-seg run --slide-path data/slides/MJUL22785295_001.mrxs
```

Run all slides in a specific directory:

```bash
uv run wsi-seg run --slide-path data/slides
```

Recursively discover slides in nested directories:

```bash
uv run wsi-seg run --slide-path data/slides --recursive
```

Isolate core pipeline timing (skip TIFF and preview exports):

```bash
uv run wsi-seg run --no-exports
```

Enable structured event logging:

```bash
uv run wsi-seg run --verbose
```

Preserve the intermediate memmap for debugging:

```bash
uv run wsi-seg run --keep-memmap
```

Override model or output directory:

```bash
uv run wsi-seg run --model-path data/other_model.pt --output-dir results
```

Inspect slide metadata and planning:

```bash
uv run wsi-seg inspect
uv run wsi-seg inspect --slide-path data/slides/MJUL22785295_001.mrxs
```

Probe TorchScript model:

```bash
uv run wsi-seg probe-model
```

When multiple slides are processed, a batch summary CSV/JSON is written under `outputs/_batches/`.

## Outputs

Each run creates its own directory under `outputs/<slide_stem>/<run_id>/`:

```text
outputs/
└── MJUL22785295_001/
    └── 20260308T131422Z_4c16605_cfg9a21f3/
        ├── mask.ome.tif          # pyramidal OME-TIFF (QuPath-compatible)
        ├── events.jsonl          # structured run events with timestamps
        ├── preview_mask.png
        ├── preview_overlay.png
        ├── preview_tissue.png    # coarse tissue mask used for scheduling
        └── run.json
```

The `run_id` is `<UTC timestamp>_<git-sha7>_<config-hash8>`, giving full traceability.
By default the temporary memmap is removed after the run; use `--keep-memmap` when you want to inspect it.
When `write_tiff: true` is set in config, a flat `mask.tif` is also written (off by default).

`run.json` includes:
- `run_id`, `started_at_utc`, `finished_at_utc`
- `git` block: commit SHA, branch, dirty flag
- slide metadata (including MPP source) and chosen level
- planning statistics with area fractions and batch stats
- wall timing + component timing breakdown with overlap semantics
- three throughput tiers: grid / ROI / candidate patches per second
- config used for the run
- coordinate mapping notes back to level-0 pixels

Batch runs write their aggregate reports under `outputs/_batches/<batch_id>/`.

## Current limitations

- It uses a coarse, conservative tissue mask heuristic rather than a learned tissue segmenter.
- Pyramid downsampling uses nearest-neighbor subsampling (`[::2, ::2]`), which is correct for binary masks but would need interpolation for probability maps.

## Suggested next steps

1. visually verify tissue masks across all three slides
2. add optional bounds-cropped intermediate/output mode
3. add manual GitHub Actions smoke test for a real slide
4. add probability blending / gaussian weighting as an optional stitcher
