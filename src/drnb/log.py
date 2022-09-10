import logging
from contextlib import contextmanager

from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger("rich")


def set_verbosity(verbose=False):
    old_level = log.getEffectiveLevel()
    if verbose:
        log.setLevel(logging.INFO)
    else:
        log.setLevel(logging.WARNING)
    return old_level


@contextmanager
def log_verbosity(verbose):
    old_log_level = set_verbosity(verbose)
    try:
        yield
    finally:
        log.setLevel(old_log_level)
