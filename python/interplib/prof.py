"""Functions to allow profiling of the code."""

import linecache
import tracemalloc
from time import perf_counter
from typing import Literal


class PerfTimer:
    """Basic performance timer."""

    prev: float

    def __init__(self) -> None:
        self.prev = perf_counter()

    def set(self) -> None:
        """Only set the timer."""
        self.prev = perf_counter()

    def stop(self, fmt: str) -> None:
        """Report time passed since last set."""
        now = perf_counter()
        print(fmt.format(now - self.prev))
        self.prev = now


class MemProfiler:
    """Memory profiling using tracemalloc."""

    was_tracing: bool
    prev_sn: tracemalloc.Snapshot

    def __init__(self, nframe: int, /) -> None:
        if tracemalloc.is_tracing():
            self.was_tracing = True
        else:
            self.was_tracing = False
            tracemalloc.start(nframe)
        self.prev_sn = tracemalloc.take_snapshot().filter_traces(
            (
                tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
                tracemalloc.Filter(False, "<unknown>"),
            )
        )

    def __del__(self) -> None:
        """Stop tracing."""
        if self.was_tracing:
            tracemalloc.stop()

    def set(self) -> None:
        """Take snapshot for future comparison."""
        self.prev_sn = tracemalloc.take_snapshot()

    def stop(
        self,
        key_type: Literal["filename", "lineno", "traceback"] = "lineno",
        limit: int = 10,
    ) -> None:
        """Get current memory differences."""
        snapshot = tracemalloc.take_snapshot().filter_traces(
            (
                tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
                tracemalloc.Filter(False, "<unknown>"),
            )
        )
        top_stats = snapshot.statistics(key_type)

        print("Top %s lines" % limit)
        for index, stat in enumerate(top_stats[:limit], 1):
            frame = stat.traceback[0]
            print(
                "#%s: %s:%s: %.1f KiB"
                % (index, frame.filename, frame.lineno, stat.size / 1024)
            )
            line = linecache.getline(frame.filename, frame.lineno).strip()
            if line:
                print("    %s" % line)

        other = top_stats[limit:]
        if other:
            size = sum(stat.size for stat in other)
            print("%s other: %.1f KiB" % (len(other), size / 1024))
        total = sum(stat.size for stat in top_stats)
        print("Total allocated size: %.1f KiB" % (total / 1024))
