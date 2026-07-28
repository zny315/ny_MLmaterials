from __future__ import annotations

import builtins
import sys
import traceback
from typing import Any


def ascii_text(value: Any) -> str:
    return str(value).encode("ascii", errors="replace").decode("ascii")


def console_print(*values: Any, **kwargs: Any) -> None:
    safe_values = tuple(ascii_text(value) for value in values)
    if "sep" in kwargs:
        kwargs["sep"] = ascii_text(kwargs["sep"])
    if "end" in kwargs:
        kwargs["end"] = ascii_text(kwargs["end"])
    builtins.print(*safe_values, **kwargs)


def install_ascii_exception_hook() -> None:
    def handle_exception(exc_type: type[BaseException], exc_value: BaseException, exc_traceback: Any) -> None:
        traceback.print_tb(exc_traceback)
        console_print(f"{exc_type.__name__}: {exc_value}", file=sys.stderr)

    sys.excepthook = handle_exception
