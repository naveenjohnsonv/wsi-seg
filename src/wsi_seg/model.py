from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from wsi_seg.utils import supports_non_blocking


@dataclass(slots=True)
class ModelProbe:
    model_path: Path
    device: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    output_kind: str


def load_torchscript_model(path: str | Path, device: torch.device) -> torch.jit.ScriptModule:
    model = torch.jit.load(str(path), map_location=str(device))
    model.eval()
    return model


def _unwrap_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    if isinstance(output, dict):
        for key in ("logits", "out", "pred", "prediction"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
        for value in output.values():
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def probe_model(path: str | Path, device: torch.device, patch_px: int) -> ModelProbe:
    model = load_torchscript_model(path, device)
    x = torch.zeros((1, 3, patch_px, patch_px), dtype=torch.float32, device=device)
    with torch.inference_mode():
        y = _unwrap_output(model(x))
    if y.ndim == 4 and y.shape[1] == 1:
        kind = "binary_logits_1ch"
    elif y.ndim == 4 and y.shape[1] == 2:
        kind = "binary_logits_2ch"
    elif y.ndim == 3:
        kind = "binary_logits_hw"
    else:
        kind = "unknown"
    return ModelProbe(
        model_path=Path(path),
        device=str(device),
        input_shape=tuple(x.shape),
        output_shape=tuple(y.shape),
        output_kind=kind,
    )


def batch_to_tensor(patches: list[np.ndarray], device: torch.device) -> torch.Tensor:
    arr = np.stack(patches, axis=0).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
    return tensor.to(device=device, non_blocking=supports_non_blocking(device))


def output_to_probs(output: Any, patch_px: int, apply_sigmoid: bool = True) -> torch.Tensor:
    y = _unwrap_output(output)
    if y.ndim == 4 and y.shape[1] == 1:
        y = y[:, 0]
        probs = torch.sigmoid(y.float()) if apply_sigmoid else y.float()
    elif y.ndim == 4 and y.shape[1] == 2:
        probs = torch.softmax(y.float(), dim=1)[:, 1]
    elif y.ndim == 3:
        probs = torch.sigmoid(y.float()) if apply_sigmoid else y.float()
    else:
        raise ValueError(f"Expected binary segmentation output; got shape {tuple(y.shape)}")

    if probs.shape[-2:] != (patch_px, patch_px):
        probs = F.interpolate(
            probs.unsqueeze(1),
            size=(patch_px, patch_px),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
    return probs
