"""Centralized logging for Quantiv (stdlib + Rich-friendly)."""

from __future__ import annotations

import logging
import sys

_configured = False


def get_logger(name: str = "quantiv") -> logging.Logger:
    """Return a configured logger (idempotent)."""
    global _configured
    logger = logging.getLogger(name)
    if not _configured:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        _configured = True
    return logger


def set_verbose(verbose: bool) -> None:
    get_logger().setLevel(logging.DEBUG if verbose else logging.INFO)
