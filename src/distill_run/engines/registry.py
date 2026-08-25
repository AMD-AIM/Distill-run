"""Lazy engine registry.

Engine modules import heavy frameworks inside their functions, and are themselves
imported only when a run needs them, so a broken trl install cannot stop a swift
run from starting.
"""

from __future__ import annotations

from typing import Final

from ..errors import UsageError
from ..params import ENGINES
from .base import Engine

_MODULES: Final[dict[str, tuple[str, str]]] = {
    "easydistill": (".easydistill", "EasyDistillEngine"),
    "swift": (".swift", "SwiftEngine"),
    "trl": (".trl", "TrlEngine"),
    "easydistill-swift": (".easydistill_swift", "EasyDistillSwiftEngine"),
    "noop": (".noop", "NoopEngine"),
}


def available() -> tuple[str, ...]:
    return tuple(_MODULES)


def load(name: str) -> Engine:
    if name not in _MODULES:
        raise UsageError(f"unknown engine '{name}'; expected one of {', '.join(ENGINES)}")
    import importlib

    module_name, class_name = _MODULES[name]
    module = importlib.import_module(module_name, package=__package__)
    return getattr(module, class_name)()
