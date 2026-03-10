from __future__ import annotations

import queue
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wsi_seg.scheduler import SuperTilePlan
from wsi_seg.slide import LevelSelection, OpenSlideReader, OutputFrame, read_output_region


@dataclass(slots=True)
class ReaderMetrics:
    active_seconds: float = 0.0
    wait_seconds: float = 0.0
    num_reads: int = 0


@dataclass(slots=True)
class PrefetchedSuperTile:
    plan: SuperTilePlan
    image: np.ndarray


@dataclass(slots=True)
class _QueueItem:
    kind: str
    payload: object | None = None
    error: Exception | None = None


class SuperTilePrefetcher:
    def __init__(
        self,
        *,
        slide_path: str | Path,
        selection: LevelSelection,
        frame: OutputFrame,
        plans: Sequence[SuperTilePlan],
        openslide_cache_bytes: int,
        queue_size: int = 2,
        mpp_override_x: float | None = None,
        mpp_override_y: float | None = None,
    ) -> None:
        self.slide_path = Path(slide_path)
        self.selection = selection
        self.frame = frame
        self.plans = list(plans)
        self.openslide_cache_bytes = openslide_cache_bytes
        self.queue_size = queue_size
        self.mpp_override_x = mpp_override_x
        self.mpp_override_y = mpp_override_y
        self.metrics = ReaderMetrics()
        self._queue: queue.Queue[_QueueItem] = queue.Queue(maxsize=max(1, queue_size))
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def __enter__(self) -> SuperTilePrefetcher:
        self._thread = threading.Thread(target=self._worker, name="wsi-seg-reader", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        return False

    def __iter__(self):
        while True:
            t0 = time.perf_counter()
            item = self._queue.get()
            wait_dt = time.perf_counter() - t0
            with self._lock:
                self.metrics.wait_seconds += wait_dt
            if item.kind == "done":
                break
            if item.kind == "error":
                assert item.error is not None
                raise item.error
            assert isinstance(item.payload, PrefetchedSuperTile)
            yield item.payload

    def _worker(self) -> None:
        try:
            with OpenSlideReader(
                self.slide_path,
                mpp_override_x=self.mpp_override_x,
                mpp_override_y=self.mpp_override_y,
            ) as slide:
                slide.set_cache(self.openslide_cache_bytes)
                for plan in self.plans:
                    t0 = time.perf_counter()
                    image = read_output_region(
                        slide,
                        self.selection,
                        frame=self.frame,
                        out_x=plan.out_x,
                        out_y=plan.out_y,
                        out_w=plan.out_w,
                        out_h=plan.out_h,
                    )
                    active_dt = time.perf_counter() - t0
                    with self._lock:
                        self.metrics.active_seconds += active_dt
                        self.metrics.num_reads += 1
                    self._queue.put(
                        _QueueItem(
                            kind="data",
                            payload=PrefetchedSuperTile(plan=plan, image=image),
                        )
                    )
                self._queue.put(_QueueItem(kind="done"))
        except Exception as exc:  # pragma: no cover
            self._queue.put(_QueueItem(kind="error", error=exc))
