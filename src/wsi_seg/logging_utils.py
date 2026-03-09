from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wsi_seg.utils import JsonEncoder


def configure_logging(verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("wsi_seg")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(rich_tracebacks=True, show_time=True)
        formatter = logging.Formatter("%(message)s")
    except Exception:  # pragma: no cover
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(handler)
    return logger


class StructuredRunLogger:
    def __init__(self, run_dir: str | Path, *, verbose: bool = False) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "events.jsonl"
        self._logger = logging.getLogger("wsi_seg")
        self.verbose = verbose

    @staticmethod
    def _ts() -> str:
        return datetime.now(UTC).isoformat()

    def event(self, name: str, **fields: Any) -> None:
        payload = {"ts": self._ts(), "event": name, **fields}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, cls=JsonEncoder) + "\n")
        if self.verbose:
            self._logger.info("%s | %s", name, fields)

    def info(self, message: str, *args: Any) -> None:
        self._logger.info(message, *args)

    def debug(self, message: str, *args: Any) -> None:
        self._logger.debug(message, *args)
