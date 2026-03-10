from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

SUPPORTED_SLIDE_SUFFIXES = (
    ".mrxs",
    ".svs",
    ".tif",
    ".tiff",
    ".ndpi",
    ".vms",
    ".vmu",
    ".scn",
    ".bif",
)


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
    out.parent.mkdir(parents=True, exist_ok=True)
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


def utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def git_info() -> dict[str, Any]:
    info: dict[str, Any] = {"git_commit": None, "git_branch": None, "git_dirty": None}
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        info["git_commit"] = sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    try:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        info["git_branch"] = branch or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    try:
        rc = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        info["git_dirty"] = rc != 0
    except FileNotFoundError:
        pass
    return info


def config_hash(cfg_dict: dict[str, Any]) -> str:
    relevant = {
        "slide": cfg_dict.get("slide", {}),
        "model": cfg_dict.get("model", {}),
        "runtime": {
            "prefetch_supertiles": cfg_dict.get("runtime", {}).get("prefetch_supertiles", True),
            "prefetch_queue_size": cfg_dict.get("runtime", {}).get("prefetch_queue_size", 2),
        },
        "schedule": cfg_dict.get("schedule", {}),
        "output": {
            "write_ome_tiff": cfg_dict.get("output", {}).get("write_ome_tiff", False),
        },
    }
    blob = json.dumps(relevant, sort_keys=True, cls=JsonEncoder).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def generate_run_id(cfg_dict: dict[str, Any]) -> str:
    ts = utc_now_iso()
    gi = git_info()
    sha = gi.get("git_commit") or "nogit"
    chash = config_hash(cfg_dict)
    return f"{ts}_{sha}_{chash}"


def _matches_suffix(path: Path, suffixes: Iterable[str]) -> bool:
    lowered = path.name.lower()
    return any(lowered.endswith(sfx.lower()) for sfx in suffixes)


def discover_slide_paths(
    path: str | Path,
    *,
    suffixes: tuple[str, ...] | None = None,
    pattern: str | None = None,
    recursive: bool = False,
) -> list[Path]:
    root = Path(path).expanduser().resolve()
    if pattern is not None:
        if root.is_file():
            if not root.match(pattern):
                return []
            return [root]
        iterator = root.rglob(pattern) if recursive else root.glob(pattern)
        return sorted(p.resolve() for p in iterator if p.is_file())

    wanted = suffixes or SUPPORTED_SLIDE_SUFFIXES
    if root.is_file():
        return [root] if _matches_suffix(root, wanted) else []

    if recursive:
        iterator = (p for p in root.rglob("*") if p.is_file())
    else:
        iterator = (p for p in root.iterdir() if p.is_file())
    return sorted(p.resolve() for p in iterator if _matches_suffix(p, wanted))
