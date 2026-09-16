"""Typing constructs that are not available on every supported interpreter."""

import sys

if sys.version_info >= (3, 12):
    from typing import override
else:  # pragma: no cover - exercised on 3.11 only
    from typing_extensions import override

__all__ = ["override"]
