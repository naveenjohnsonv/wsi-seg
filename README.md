



# WSI Segmentation Pipeline

Memory-efficient whole-slide inference pipeline for binary lesion masking on gigapixel pathology slides.

---

## Overview

Whole-slide images (WSIs) are multi-resolution pathology scans that can be hundreds of thousands of pixels wide and tall. Running a segmentation model on them requires more than just model execution - it demands careful management of pyramid level selection, I/O scheduling, patch stitching, and memory.

This pipeline runs a pre-trained TorchScript model that expects **512×512 RGB patches at 0.88 µm/px**, normalized to `[0, 1]`, and returns per-pixel binary segmentation scores. It keeps memory bounded, makes I/O choices explicit, and produces viewer-friendly outputs (pyramidal OME-TIFF masks, preview overlays, and structured run manifests).

---

## Architecture

```text
MRXS WSI
  → inspect metadata (MPP, levels, bounds)
  → choose nearest pyramid level to target MPP
  → build conservative coarse tissue mask from thumbnail
  → schedule candidate output-space patches inside bounds/tissue
  → group patches into supertiles
  → read one supertile at a time from OpenSlide
  → resize once to exact target MPP
  → split into overlapping 512×512 patches
  → batched TorchScript inference
  → overlap + center-crop stitching
  → disk-backed mask writer (numpy memmap)
  → export OME-TIFF + previews + run manifest
```

### Key Design Decisions

**Physical-scale policy.** The target MPP (0.88 µm/px) is authoritative. The pipeline selects the pyramid level with the nearest MPP, then applies a single resize to match exactly. `mpp_x` and `mpp_y` are tracked independently. This reduces decode volume versus always reading level 0.

**I/O scheduling.** Slide bounds and a conservative thumbnail-level tissue mask prune obvious background. Candidate patches are grouped into output-space **supertiles** (default 4096 px) so that OpenSlide reads are large and spatially local rather than thousands of tiny random accesses. An optional prefetch thread overlaps reads with inference.

**Stitching.** Patches are inferred with overlap (stride 384, halo 64). Only the valid center region of each patch is written back, eliminating most border artifacts without needing a full-resolution float accumulator.

**Storage.** The intermediate mask is written to a `numpy.memmap` on disk, bounding peak RAM regardless of slide size. The final artifact is a pyramidal OME-TIFF for review in standard image tools.

**Backend.** OpenSlide is used for MRXS/MIRAX reading for its maturity and reliability with this format.

### Scaling Vision

The current architecture is designed with a clear path to production scale. Supertile-based scheduling and the memmap intermediate naturally extend to **Zarr chunked storage** for cloud-native random writes. The prefetch thread can be replaced by a **multi-worker DataLoader** pattern for higher GPU utilization. The per-slide pipeline is stateless, making **distributed multi-slide execution** straightforward (e.g., one slide per worker in a task queue). Additional planned improvements include Gaussian blending / importance-map merging for smoother probability maps, bounds-cropped output mode to reduce artifact size, and upstream artifact/QC masking before lesion inference.

---

## Repository Layout

```text
.
├── configs/
│   └── default.yaml
├── src/wsi_seg/
│   ├── cli.py
│   ├── config.py
│   ├── geometry.py
│   ├── logging_utils.py
│   ├── model.py
│   ├── pipeline.py
│   ├── prefetch.py
│   ├── preview.py
│   ├── scheduler.py
│   ├── slide.py
│   ├── tissue.py
│   ├── utils.py
│   └── writer.py
├── tests/
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## Setup

### Local with `uv`

```bash
uv sync --extra cpu --group dev
```

Other supported extras in `pyproject.toml`:

```bash
uv sync --extra cu128 --group dev   # CUDA 12.8
uv sync --extra xpu --group dev     # Intel XPU
uv sync --extra rocm --group dev    # AMD ROCm
```

### Docker

```bash
docker build -t wsi-seg .
```

The Docker image installs the **CPU** extra by default for reproducibility.

### Data Layout

Place the model and slides as follows:

```text
data/
├── model_scripted.pt
└── slides/
    ├── MJUL22785295_001.mrxs
    ├── MJUL22785295_001/
    ├── MKQD63856403_001.mrxs
    ├── MKQD63856403_001/
    ├── QOFN21275156_001.mrxs
    └── QOFN21275156_001/
```

---

## Usage

### Inspect a slide

```bash
uv run wsi-seg inspect --slide-path data/slides/MJUL22785295_001.mrxs
```

Prints vendor, level pyramid, level-0 MPP, chosen inference level, output mask size, bounds-aware ROI, and candidate patch/supertile counts.

### Probe the TorchScript model

```bash
uv run wsi-seg probe-model --model-path data/model_scripted.pt
```

### Run inference

```bash
# Single slide
uv run wsi-seg run --slide-path data/slides/MJUL22785295_001.mrxs

# All slides in a directory
uv run wsi-seg run --slide-path data/slides

# Recursive directory search
uv run wsi-seg run --slide-path data/slides --recursive

# Skip export artifacts (isolate core pipeline timing)
uv run wsi-seg run --slide-path data/slides/QOFN21275156_001.mrxs --no-exports
```

---

## Configuration

Main knobs are exposed in `configs/default.yaml`:

```yaml
model:
  target_mpp: 0.88
  patch_px: 512
  stride_px: 384
  halo_px: 64
  batch_size: 8
  threshold: 0.5

runtime:
  device: auto
  use_amp: true
  openslide_cache_bytes: 536870912
  prefetch_supertiles: true
  prefetch_queue_size: 2

schedule:
  use_bounds: true
  use_tissue_mask: true
  tissue_mask_min_fraction: 0.03
  supertile_px: 4096

output:
  write_ome_tiff: true
  write_tiff: false
  write_previews: true
  keep_memmap: false
```

---

## Outputs

Each run writes to `outputs/<slide_stem>/<run_id>/`:

```text
mask.ome.tif          # pyramidal binary mask
preview_mask.png      # binary preview of final prediction
preview_overlay.png   # slide thumbnail + green predicted mask
preview_tissue.png    # coarse tissue mask used for scheduling
events.jsonl          # structured run events
run.json              # manifest with metadata, timings, config, and planning stats
```

`run.json` includes slide metadata and bounds, chosen level and output shape, planning statistics, timing breakdown, throughput metrics, config snapshot, and coordinate mapping back to level-0 pixels.

---

## Visual Results

<!-- Add result images under docs/results/ and replace the inline code below with actual Markdown image tags. -->

```text
docs/
└── results/
    ├── MJUL22785295_001_thumb.png
    ├── MJUL22785295_001_tissue.png
    ├── MJUL22785295_001_mask.png
    ├── MJUL22785295_001_overlay.png
    ├── MKQD63856403_001_thumb.png
    ├── MKQD63856403_001_tissue.png
    ├── MKQD63856403_001_mask.png
    ├── MKQD63856403_001_overlay.png
    ├── QOFN21275156_001_thumb.png
    ├── QOFN21275156_001_tissue.png
    ├── QOFN21275156_001_mask.png
    └── QOFN21275156_001_overlay.png
```

| Slide | Tissue mask | Binary mask | Overlay |
|---|---|---|---|
| MJUL22785295_001 | `![MJUL tissue](docs/results/MJUL22785295_001_tissue.png)` | `![MJUL mask](docs/results/MJUL22785295_001_mask.png)` | `![MJUL overlay](docs/results/MJUL22785295_001_overlay.png)` |
| MKQD63856403_001 | `![MKQD tissue](docs/results/MKQD63856403_001_tissue.png)` | `![MKQD mask](docs/results/MKQD63856403_001_mask.png)` | `![MKQD overlay](docs/results/MKQD63856403_001_overlay.png)` |
| QOFN21275156_001 | `![QOFN tissue](docs/results/QOFN21275156_001_tissue.png)` | `![QOFN mask](docs/results/QOFN21275156_001_mask.png)` | `![QOFN overlay](docs/results/QOFN21275156_001_overlay.png)` |

> Replace the inline code above with actual Markdown image tags once the assets are committed.

---

## Performance

Comparison between earlier end-to-end runs and the current architecture (both with preview and OME-TIFF export enabled):

| Slide | End-to-end wall time | Core processing time | Candidate patches/sec |
|---|---:|---:|---:|
| MJUL22785295_001 | 88.2 s → 69.4 s (**−21.3%**) | 57.6 s → 40.3 s (**−30.1%**) | 11.0 → 13.9 (**+27.0%**) |
| MKQD63856403_001 | 179.2 s → 117.5 s (**−34.4%**) | 145.4 s → 88.2 s (**−39.3%**) | 16.2 → 24.6 (**+52.4%**) |
| QOFN21275156_001 | 71.5 s → 48.7 s (**−31.9%**) | 40.9 s → 20.0 s (**−51.1%**) | 8.7 → 12.8 (**+46.9%**) |

On average: **~29%** faster end-to-end, **~40%** faster core processing, and **~42%** higher patch throughput. The gains come primarily from bounded prefetch between read and infer stages, supertile-based reads improving locality, and reduced critical-path reader stalls.

---

## Testing

The test suite includes config validation tests, slide metadata extraction tests, scheduling and crop-geometry tests, and writer tests for TIFF/OME-TIFF metadata.

---

## Trade-offs and Known Limitations

**Chosen trade-offs.** OpenSlide was selected over more experimental readers for MRXS reliability. Center-crop stitching was chosen over weighted probability blending for simpler bounded-memory writeback. A numpy memmap intermediate was preferred over Zarr for lower implementation risk. The optional single prefetch thread was favored over a more complex multi-worker DataLoader design.

**Known limitations.** The coarse tissue mask is heuristic and conservative by design - it may include some background regions but avoids missing tissue. Current CI uses synthetic tests rather than full real-slide runs.