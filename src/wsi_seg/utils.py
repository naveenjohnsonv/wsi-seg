from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


class JsonEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, torch.device):
            return str(obj)
        return super().default(obj)


def dump_json(data: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=JsonEncoder)


_ASYNC_TRANSFER_DEVICES = {"cuda", "xpu"}
_AMP_DEVICES = {"cuda", "xpu"}


def supports_non_blocking(device: torch.device) -> bool:
    return device.type in _ASYNC_TRANSFER_DEVICES


def supports_amp(device: torch.device) -> bool:
    return device.type in _AMP_DEVICES


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch, "xpu", None) and torch.xpu.is_available():
        return torch.device("xpu")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(requested: str) -> torch.device:
    req = requested.strip().lower()
    if req == "auto":
        return _auto_device()
    if req.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    if req.startswith("xpu") and getattr(torch, "xpu", None) and torch.xpu.is_available():
        return torch.device(requested)
    if req == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def format_bytes(n_bytes: int) -> str:
    suffixes = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(n_bytes)
    for suffix in suffixes:
        if value < 1024 or suffix == suffixes[-1]:
            return f"{value:.2f} {suffix}"
        value /= 1024.0
    return f"{value:.2f} TiB"
