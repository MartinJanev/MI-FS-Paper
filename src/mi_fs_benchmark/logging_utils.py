from __future__ import annotations

import logging
import sys
from typing import Optional


class TqdmLoggingHandler(logging.Handler):
    """Custom logging handler that writes through tqdm to avoid progress bar interference."""

    def emit(self, record):
        try:
            from tqdm import tqdm

            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
        except Exception:
            self.handleError(record)


def setup_logging(level: int = logging.INFO, use_tqdm: bool = True, concise: bool = True) -> None:
    """
    Configure global logging for scripts and library code.

    Parameters
    ----------
    level:
        Logging level, e.g. logging.INFO or logging.DEBUG.
    use_tqdm:
        If True, use TqdmLoggingHandler to avoid interfering with progress bars.
    concise:
        If True, use a compact log format focused on message content.
    """
    # Remove any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    if use_tqdm:
        handler = TqdmLoggingHandler()
    else:
        handler = logging.StreamHandler(sys.stderr)

    if concise:
        formatter = logging.Formatter(fmt="%(levelname)s: %(message)s")
    else:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)
    root_logger.setLevel(level)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger.

    Parameters
    ----------
    name:
        Logger name, typically __name__ of the caller.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(name)
