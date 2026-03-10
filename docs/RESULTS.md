# Results

The images below show the latest segmentation results for four slides.

## Available result files

```text
docs/
└── results/
    ├── KLMP45690052_001_tissue.png
    ├── KLMP45690052_001_mask.png
    ├── KLMP45690052_001_overlay.png
    ├── MJUL22785295_001_tissue.png
    ├── MJUL22785295_001_mask.png
    ├── MJUL22785295_001_overlay.png
    ├── MKQD63856403_001_tissue.png
    ├── MKQD63856403_001_mask.png
    ├── MKQD63856403_001_overlay.png
    ├── QOFN21275156_001_tissue.png
    ├── QOFN21275156_001_mask.png
    └── QOFN21275156_001_overlay.png
```

## Results table

| Slide ID | Tissue mask | Binary mask | Overlay |
|---|---|---|---|
| KLMP45690052_001 | ![KLMP45690052_001 tissue](results/KLMP45690052_001_tissue.png) | ![KLMP45690052_001 mask](results/KLMP45690052_001_mask.png) | ![KLMP45690052_001 overlay](results/KLMP45690052_001_overlay.png) |
| MJUL22785295_001 | ![MJUL22785295_001 tissue](results/MJUL22785295_001_tissue.png) | ![MJUL22785295_001 mask](results/MJUL22785295_001_mask.png) | ![MJUL22785295_001 overlay](results/MJUL22785295_001_overlay.png) |
| MKQD63856403_001 | ![MKQD63856403_001 tissue](results/MKQD63856403_001_tissue.png) | ![MKQD63856403_001 mask](results/MKQD63856403_001_mask.png) | ![MKQD63856403_001 overlay](results/MKQD63856403_001_overlay.png) |
| QOFN21275156_001 | ![QOFN21275156_001 tissue](results/QOFN21275156_001_tissue.png) | ![QOFN21275156_001 mask](results/QOFN21275156_001_mask.png) | ![QOFN21275156_001 overlay](results/QOFN21275156_001_overlay.png) |

## What the artifacts mean

- **Tissue mask**: a thumbnail-derived heuristic used only for scheduling
- **Binary mask**: the final thresholded model prediction
- **Overlay**: the slide thumbnail with the binary mask overlaid

## Iteration improvements

The table below tracks the same slide workloads across three pipeline iterations on the current hardware. Each row shows the first run, an intermediate run, and the latest manifest-backed run; the bold percentage is the net change from the first iteration to the latest one.

**Comparison basis**
- first: earlier end-to-end runs with preview export enabled
- second: intermediate iteration runs with preview export enabled and OME-TIFF export enabled
- latest: newest manifest-backed runs currently present under `outputs/`
- note: the bold percentage in each row is the net change from **first -> latest**
- note: export payloads are not identical, so the cleanest architecture comparison is the **core processing** row

| Slide | End-to-end wall time | Core processing time* | Candidate patches/sec |
|---|---:|---:|---:|
| MJUL22785295_001 | 88.2s → 69.4s → 53.7s (**-39.1%**) | 57.6s → 40.3s → 47.4s (**-17.8%**) | 11.0 → 13.9 → 18.3 (**+66.3%**) |
| MKQD63856403_001 | 179.2s → 117.5s → 137.3s (**-23.4%**) | 145.4s → 88.2s → 130.3s (**-10.4%**) | 16.2 → 24.6 → 21.0 (**+29.9%**) |
| QOFN21275156_001 | 71.5s → 48.7s → 33.7s (**-52.9%**) | 40.9s → 20.0s → 30.4s (**-25.6%**) | 8.7 → 12.8 → 18.5 (**+112.1%**) |

\* first-iteration core processing = `read_supertiles + model_infer + writeback`; second and latest core processing = `processing_loop`

### Main takeaways

- average **end-to-end wall time** improved by about **39%**
- average **core processing time** improved by about **18%**
- average **candidate-patch throughput** improved by about **69%**
- overall improvement in times despite newer runs using `prefer_higher_bounded`, which read finer native levels before resizing to the target MPP, increasing compute time