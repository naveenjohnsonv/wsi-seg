# Results Placeholders

Add result images under `docs/results/` and replace the inline code blocks below with real Markdown image tags when assets are committed.

## Suggested directory layout

```text
docs/
└── results/
    ├── slide_1_thumb.png
    ├── slide_1_tissue.png
    ├── slide_1_mask.png
    ├── slide_1_overlay.png
    ├── slide_2_thumb.png
    ├── slide_2_tissue.png
    ├── slide_2_mask.png
    ├── slide_2_overlay.png
    ├── slide_3_thumb.png
    ├── slide_3_tissue.png
    ├── slide_3_mask.png
    └── slide_3_overlay.png
```

## Placeholder table

| Slide | Tissue mask | Binary mask | Overlay |
|---|---|---|---|
| slide_1 | `![slide_1 tissue](docs/results/slide_1_tissue.png)` | `![slide_1 mask](docs/results/slide_1_mask.png)` | `![slide_1 overlay](docs/results/slide_1_overlay.png)` |
| slide_2 | `![slide_2 tissue](docs/results/slide_2_tissue.png)` | `![slide_2 mask](docs/results/slide_2_mask.png)` | `![slide_2 overlay](docs/results/slide_2_overlay.png)` |
| slide_3 | `![slide_3 tissue](docs/results/slide_3_tissue.png)` | `![slide_3 mask](docs/results/slide_3_mask.png)` | `![slide_3 overlay](docs/results/slide_3_overlay.png)` |

## What the artifacts mean

- **Tissue mask**: a thumbnail-derived heuristic used only for scheduling
- **Binary mask**: the final thresholded model prediction
- **Overlay**: the slide thumbnail with the binary mask overlaid