# WSI Segmentation Pipeline

A production-style whole-slide image (WSI) segmentation pipeline built on **OpenSlide** and **PyTorch**.

The pipeline performs lesion segmentation on MIRAX/MRXS slides at a target physical resolution (e.g. **0.88 µm/px**), using a TorchScript model and a memory-safe, patch-based inference loop. It is designed for clarity, reproducibility, and easy extension with more advanced I/O and scheduling strategies.

This first implementation is intentionally a **working baseline**:

- inspect slide metadata and pyramid levels
- probe the TorchScript segmentation model contract
- run end-to-end inference on one slide at the target physical resolution
- write a disk-backed binary mask
- export a TIFF mask, a preview overlay, and a `run.json` manifest

The next pass should add:

- conservative coarse tissue pruning
- supertile-based reading
- optional reader prefetch
- richer benchmarking and tests

## Why this baseline exists

The point of the first implementation is to lock down the hard correctness pieces first:

- physical-scale logic (`target_mpp`)
- level choice and resizing
- TorchScript input/output assumptions
- coordinate transforms between output mask pixels and level-0 pixels
- bounded memory usage via memmap
- reproducible outputs and logging

## Chosen architecture for v0

- backend: **OpenSlide**
- scale policy: **target MPP is authoritative**
- read-level policy: **nearest level by MPP error, then resize**
- inference granularity: **patch-by-patch baseline**
- stitching: **overlap + center-crop writeback**
- intermediate storage: **NumPy memmap**
- final outputs: **TIFF + preview overlay + run manifest**

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
│       ├── model.py
│       ├── pipeline.py
│       ├── preview.py
│       ├── slide.py
│       ├── utils.py
│       └── writer.py
├── tests/
│   ├── test_config.py
│   └── test_pipeline_math.py
├── .gitignore
├── Dockerfile
├── pyproject.toml
└── README.md
└── uv.lock
```

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then sync with the extra that matches your GPU:

```bash
uv sync --extra xpu   --group dev   # Intel Arc / Xe
uv sync --extra cu128 --group dev   # NVIDIA CUDA 12.8
uv sync --extra rocm  --group dev   # AMD ROCm 6.4
uv sync --extra cpu   --group dev   # CPU only
```

### Verify GPU detection

```bash
# Intel Arc / Xe
uv run python -c "import torch; print(torch.__version__, torch.xpu.is_available())"

# NVIDIA
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Docker

```bash
docker build -t wsi-seg .
docker run --rm -it -v "$PWD":/app wsi-seg inspect configs/default.yaml
```

## Data layout

```text
data/
├── model_scripted.pt
└── slides/
    ├── MJUL22785295_001.mrxs
    └── MJUL22785295_001/
    └── ...
```

## Commands

### Inspect a slide

```bash
wsi-seg inspect configs/default.yaml
```

Prints slide metadata, pyramid levels, chosen level, and estimated output size.

### Probe the model

```bash
wsi-seg probe-model configs/default.yaml
```

Loads the TorchScript model and prints input/output shapes and output kind.

### Run the pipeline

```bash
wsi-seg run configs/default.yaml
```

Runs end-to-end inference and writes outputs to `paths.output_dir`:

```text
outputs/example_run/
├── mask.tmp.npy       # intermediate memmap
├── mask.tif           # final binary mask
├── preview_mask.png
├── preview_overlay.png
└── run.json           # run manifest with metadata, timings, config
```

`run.json` includes:

- slide metadata
- chosen level and level MPP
- output mask shape
- config used for the run
- timing summary
- coordinate mapping notes

## Current limitations

- no tissue masking yet
- no supertiles yet
- no reader prefetch yet
- default TIFF export is simple, not pyramidal
- assumes binary segmentation output

## Suggested next commits

1. add coarse tissue/background mask
2. switch patch reads to supertile reads
3. add optional OpenSlide cache configuration
4. add benchmark command and richer timings
5. add manual GitHub Actions smoke test for a real slide
