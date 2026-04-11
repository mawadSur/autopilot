from __future__ import annotations

import builtins
import sys
from typing import Optional
from loguru import logger

_DEFAULT_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}"
_ORIG_PRINT = builtins.print
_IS_PATCHED = False


def setup_logging(level: str = "INFO", *, serialize: bool = False, patch_print: bool = True) -> None:
    """Configure loguru and optionally route print() to logger.info."""
    logger.remove()
    logger.add(
        sink=sys.stdout,
        level=level.upper(),
        format=None if serialize else _DEFAULT_FORMAT,
        serialize=serialize,
    )

    def _patched_print(*args, **kwargs):
        sep = kwargs.get("sep", " ")
        msg = sep.join(str(x) for x in args)
        logger.opt(depth=1).info(msg)

    global _IS_PATCHED
    if patch_print and not _IS_PATCHED:
        builtins.print = _patched_print
        _IS_PATCHED = True


def restore_print() -> None:
    global _IS_PATCHED
    builtins.print = _ORIG_PRINT
    _IS_PATCHED = False


__all__ = ["logger", "setup_logging", "restore_print"]
