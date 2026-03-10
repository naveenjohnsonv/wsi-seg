# Running Notes

## Native runtimes

CPU:

```bash
uv sync --extra cpu --group dev
uv run wsi-seg run configs/default.yaml --slide-path data/slides/<slide>.mrxs
```

Intel XPU:

```bash
uv sync --extra xpu --group dev
uv run wsi-seg run configs/default.yaml --slide-path data/slides/<slide>.mrxs
```

NVIDIA CUDA:

```bash
uv sync --extra cu128 --group dev
uv run wsi-seg run configs/default.yaml --slide-path data/slides/<slide>.mrxs
```

AMD ROCm:

```bash
uv sync --extra rocm --group dev
uv run wsi-seg run configs/default.yaml --slide-path data/slides/<slide>.mrxs
```

## Docker

### Intel XPU (Arc / integrated GPU)

Before building, edit the `RUN uv sync` line in the Dockerfile to use
`--extra xpu` instead of `--extra cpu`.

```bash
docker build -t wsi-seg-xpu .
```

Requires Intel GPU drivers installed on the host with `/dev/dxg` available.

```bash
docker run --rm \
  --device /dev/dxg \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -v /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
  -v "/path/to/data:/app/data" \
  -v "/path/to/outputs:/app/outputs" \
  wsi-seg-xpu run /app/configs/default.yaml \
  --slide-path /app/data/slides/<slide>.mrxs
```

### NVIDIA CUDA

Before building, edit the `RUN uv sync` line in the Dockerfile to use
`--extra cu124` instead of `--extra cpu`.

```bash
docker build -t wsi-seg-cuda .
```

Requires [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed on the host.

```bash
docker run --rm \
  --gpus all \
  -v "/path/to/data:/app/data" \
  -v "/path/to/outputs:/app/outputs" \
  wsi-seg-cuda run /app/configs/default.yaml \
  --slide-path /app/data/slides/<slide>.mrxs
```

### macOS

Docker runs in a Linux VM and will not expose Apple MPS. For Apple GPU use, prefer a native `uv` install.

## CLI commands

Inspect:

```bash
uv run wsi-seg inspect configs/default.yaml --slide-path data/slides/<slide>.mrxs
```

Probe model:

```bash
uv run wsi-seg probe-model configs/default.yaml --model-path data/model_scripted.pt
```

Run one slide:

```bash
uv run wsi-seg run configs/default.yaml --slide-path data/slides/<slide>.mrxs
```

Run a directory:

```bash
uv run wsi-seg run configs/default.yaml --slide-path data/slides
```

Run recursively:

```bash
uv run wsi-seg run configs/default.yaml --slide-path data/slides --recursive
```

Disable exports:

```bash
uv run wsi-seg run configs/default.yaml --slide-path data/slides/<slide>.mrxs --no-exports
```

Keep the intermediate memmap:

```bash
uv run wsi-seg run configs/default.yaml --slide-path data/slides/<slide>.mrxs --keep-memmap
```

## Supported input discovery

The CLI discovers slide files by suffix and currently accepts:

- `.mrxs`
- `.svs`
- `.tif`
- `.tiff`
- `.ndpi`
- `.vms`
- `.vmu`
- `.scn`
- `.bif`

## Practical notes

### Low-resource environments

The pipeline has been validated under constrained settings (batch size 1, 64 MiB OpenSlide cache, prefetch disabled, CPU-only, inside Docker). It completes without requiring full-slide in-memory allocation.

### Configuration presets

These are not just the knobs to change in `configs/default.yaml` depending on the use case.

| Knob | Fast local test | Full export run |
|---|---|---|
| `runtime.device` | `cpu` | `auto` |
| `runtime.use_amp` | `false` | `true` |
| `runtime.prefetch_supertiles` | `false` | `true` |
| `runtime.prefetch_queue_size` | `1` | `6` |
| `runtime.openslide_cache_bytes` | `67108864` (64 MiB) | `536870912` (512 MiB) |
| `model.batch_size` | `1` | `8` |
| `schedule.supertile_px` | `2048` | `4096` |
| `output.write_ome_tiff` | `false` | `true` |
| `output.write_previews` | `true` | `true` |
| CLI flag | `--no-exports` | (default) |

The fast preset minimizes memory pressure and skips expensive export steps. It is useful for verifying that the pipeline completes on a new slide or environment. The full preset enables all optimizations and produces the complete artifact set.

### Config path behavior

The first CLI argument is the YAML config path. If omitted, it defaults to:

```text
configs/default.yaml
```

Relative paths inside the config are resolved either from the config directory or from the current working directory, depending on how they are declared.

### MPP overrides

If a slide is missing reliable physical-resolution metadata, set:

```yaml
slide:
  mpp_override_x: ...
  mpp_override_y: ...
```

If only one axis is provided, it is mirrored to the other axis.

### Output mapping

The exported mask may represent the full slide or a bounds-local output frame. Use `run.json` to recover the mapping back to level-0 coordinates through:

- `output_frame.origin_x_level0`
- `output_frame.origin_y_level0`
- `output_frame.width_level0`
- `output_frame.height_level0`
- `output_frame.actual_output_mpp_x`
- `output_frame.actual_output_mpp_y`