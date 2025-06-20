"""Really cool looking progress bar type."""

from dataclasses import dataclass

from numpy import log

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
                (log(self.err_initial) - log(self.err_cur))
                / (log(self.err_initial) - log(self.err_tol))
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


# if __name__ == "__main__":
#     from time import sleep

#     tracker = ProgressTracker(1e-5, 1, 1, 10, iter_width=20, err_width=20)

#     fake_errors = (1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1, 10, 100, 1e-50)

#     for i in range(len(fake_errors)):
#         tracker.update_iteration(fake_errors[i])
#         print(tracker.state_str("{} {} {}"), end="\r")
#         sleep(0.25)
