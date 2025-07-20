"""Really cool looking progress bar type."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

_ANSI_RESET = "\033[0m"
_ANSI_RED = "\033[31m"
_ANSI_YELLOW = "\033[33m"
_ANSI_GREEN = "\033[32m"

_SPINNERS = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")  # ("◐", "◓", "◑", "◒")


@dataclass
class ProgressTracker:
    """Type for progress tracking."""

    err_tol: float
    err_initial: float
    err_cur: float

    iter_max: int

    iter_cur: int = 0

    iter_width: int = 10
    err_width: int = 10

    def update_iteration(self, new_err: float) -> None:
        """Update with the new iteration error."""
        self.err_cur = new_err
        if self.iter_cur < self.iter_max:
            self.iter_cur += 1

    @property
    def iter_progress(self) -> str:
        """Return iteration progress string."""
        frac_progress = int(self.iter_width * self.iter_cur / self.iter_max)
        return "█" * frac_progress + " " * (self.iter_width - frac_progress)

    @property
    def err_indicator(self) -> str:
        """Return error indicator string."""
        err_str_list = list(c for c in " |" + " " * self.err_width + "| ")
        if self.err_cur >= self.err_initial:
            err_str_list = [_ANSI_RED + "*" + _ANSI_RESET] + err_str_list[1:]
        elif self.err_cur <= self.err_tol:
            err_str_list = err_str_list[:-1] + [_ANSI_GREEN + "*" + _ANSI_RESET]
        else:
            err_scale = int(
                (np.log(self.err_initial) - np.log(self.err_cur))
                / (np.log(self.err_initial) - np.log(self.err_tol))
                * self.err_width
            )
            assert err_scale < self.err_width and err_scale >= 0
            err_str_list[err_scale + 2] = "*"
            err_str_list = (
                err_str_list[: err_scale + 2]
                + [_ANSI_YELLOW + "*" + _ANSI_RESET]
                + err_str_list[err_scale + 3 :]
            )

        return "".join(err_str_list)

    def state_str(self, format_string: str) -> str:
        """Get a string with current state.

        Parameters
        ----------
        format_string : str
            String that will be formatted with :meth:`str.format` method.
            It will receive the spinner character, the iteration progress bar
            and error indicator bar in that order.

        Returns
        -------
        str
            Formatted string.
        """
        iter_str = (
            "Iteration "
            + str(self.iter_cur).rjust(len(str(self.iter_max)))
            + " out of "
            + str(self.iter_max)
            + "["
            + self.iter_progress
            + "]"
        )

        err_str = (
            f"Error at {self.err_cur:.3e} / {self.err_tol:.3e} ["
            + self.err_indicator
            + "]"
        )

        return format_string.format(
            _SPINNERS[self.iter_cur % len(_SPINNERS)], iter_str, err_str
        )


@dataclass(frozen=True)
class HistogramFormat:
    """Type used to format a histogram from an array."""

    rows: int
    cols: int
    tick_count: int = 2
    label_format: Callable[[float], str] = str

    def format(self, a: npt.ArrayLike) -> str:
        """Create a histogram string based on the array."""
        # Compute histogram and scaling
        hist, bin_edges = np.histogram(a, bins=self.cols)
        max_val = hist.max()
        if max_val == 0:
            scaled = np.zeros_like(hist)
        else:
            scaled = np.round((hist / max_val) * (self.rows - 1)).astype(int)

        # Print bars
        for r in reversed(range(self.rows)):
            print("".join("█" if scaled[b] >= r else " " for b in range(self.cols)))

        # Tick positions
        tick_positions = np.linspace(0, self.cols - 1, self.tick_count, dtype=int)
        tick_row = [" " for _ in range(self.cols)]
        for pos in tick_positions:
            tick_row[pos] = "|"
        print("".join(tick_row))

        # Label row with adjusted alignment
        label_row = [" " for _ in range(self.cols)]
        for i, pos in enumerate(tick_positions):
            label = self.label_format(float(bin_edges[pos]))
            label_len = len(label)
            # First tick: left align
            if i == 0:
                start = pos
            # Last tick: right align
            elif i == len(tick_positions) - 1:
                start = pos - label_len + 1
            # Middle ticks: center
            else:
                start = pos - label_len // 2
            # Boundary checks
            start = max(start, 0)
            end = min(start + label_len, self.cols)
            label_to_insert = label[: end - start]
            # Place label
            for j, ch in enumerate(label_to_insert):
                if 0 <= start + j < self.cols:
                    label_row[start + j] = ch
        return "".join(label_row)

    def __call__(self, a: npt.ArrayLike) -> str:
        """Call ``self.format(a)``."""
        return self.format(a)
