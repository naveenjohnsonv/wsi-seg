# WSI Segmentation Pipeline

A production-style whole-slide image (WSI) lesion-segmentation inference pipeline built on **OpenSlide** and **PyTorch**.

It is designed around the constraints of digital pathology:

- **massive** multi-resolution whole-slide images
- model inputs defined in **physical resolution** (`target_mpp`), not just nominal magnification
- **bounded memory** via disk-backed intermediate outputs
- **I/O-aware scheduling** through bounds-aware planning, conservative coarse tissue masking, and supertile reads

## Current implemented phases

### Phase 0 — project scaffold
- CLI (`inspect`, `probe-model`, `run`, `benchmark`)
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

## Chosen architecture

- backend: **OpenSlide**
- scale policy: **target MPP is authoritative**
- read level: **nearest level by MPP error, then resize once**
- schedule ROI: **slide bounds projected to output space**
- pruning: **conservative coarse tissue mask**
- read granularity: **supertile reads**
- stitching: **overlap + center-crop writeback**
- intermediate storage: **NumPy memmap**
- final outputs: **TIFF + previews + run manifest**

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
│       ├── model.py
│       ├── pipeline.py
│       ├── preview.py
│       ├── scheduler.py
│       ├── slide.py
│       ├── tissue.py
│       ├── utils.py
│       └── writer.py
├── tests/
│   ├── test_config.py
│   ├── test_pipeline_math.py
│   └── test_scheduler.py
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

## Data layout

```text
data/
├── model_scripted.pt
└── slides/
    ├── MJUL22785295_001.mrxs
    └── MJUL22785295_001/
```

## Commands

### Inspect a slide and planned schedule

```bash
uv run wsi-seg inspect configs/default.yaml
```

Prints:
- pyramid metadata
- chosen inference level
- output mask size
- bounds-projected ROI
- grid patch count / ROI patch count / candidate patch count / supertile count

### Probe the TorchScript model

```bash
uv run wsi-seg probe-model configs/default.yaml
```

### Run the pipeline

```bash
uv run wsi-seg run configs/default.yaml
```

### Run with the same pipeline but print throughput-oriented stats

```bash
uv run wsi-seg benchmark configs/default.yaml
```

## Outputs

```text
outputs/example_run/
├── mask.tmp.npy          # intermediate memmap
├── mask.tif              # final binary mask
├── preview_mask.png
├── preview_overlay.png
├── preview_tissue.png    # coarse tissue mask used for scheduling
└── run.json
```

`run.json` includes:
- slide metadata
- chosen level and physical resolution
- planning statistics
- timing and throughput
- config used for the run
- coordinate mapping notes back to level-0 pixels

## Current trade-offs

This version intentionally prioritizes **clarity and correctness** over the last bit of performance.

- It still writes a **full-size output mask** at the target MPP.
- It uses a **simple thumbnail-based tissue mask**, not a learned foreground model.
- It uses **synchronous supertile execution**; reader prefetch is a future improvement.
- It exports a simple TIFF mask, not yet a pyramidal OME-TIFF.

## Suggested next commits after this version

1. add buffered reader prefetch for supertiles
2. add optional bounds-cropped intermediate/output mode
3. add richer benchmark timing breakdowns per stage
4. add probability blending / gaussian weighting as an optional stitcher
