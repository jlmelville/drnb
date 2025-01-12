import logging
from contextlib import contextmanager
from typing import Any, Generator

import rich
from rich.console import Console
from rich.logging import RichHandler

# https://github.com/Textualize/rich/issues/3335
rich.jupyter.JUPYTER_HTML_FORMAT = """\
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace;margin-bottom:0px;margin-top:0px">{code}</pre>
"""
console = Console(width=100)
FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(console=console)],
)
log = logging.getLogger("rich")


def set_verbosity(verbose: bool = False) -> int:
    """Set the verbosity of the logger to INFO if `verbose` is True, else WARNING.
    Returns the old log level."""
    old_level = log.getEffectiveLevel()
    if verbose:
        log.setLevel(logging.INFO)
    else:
        log.setLevel(logging.WARNING)
    return old_level


@contextmanager
def log_verbosity(verbose: bool = False) -> Generator[None, Any, None]:
    """Context manager to temporarily set the verbosity of the logger."""
    old_log_level = set_verbosity(verbose=verbose)
    try:
        yield
    finally:
        log.setLevel(old_log_level)


def is_progress_report_iter(
    it: int, total_iterations: int, max_progress_updates: int = 10
) -> bool:
    """Check if the current iteration is a progress report iteration.
    Maximum number of progress updates is capped at `max_progress_updates`."""
    progress_frequency = total_iterations // max_progress_updates
    return (
        progress_frequency != 0
        and it % progress_frequency == 0
        and it // progress_frequency < max_progress_updates
    )


def log_progress(it: int, total_iterations: int, max_progress_updates: int = 10):
    """Log progress of a loop iteration, if it is a progress report iteration.
    Maximum number of progress updates is capped at `max_progress_updates`."""
    if is_progress_report_iter(it, total_iterations, max_progress_updates):
        log.info("permutation run %d completed", it + 1)
