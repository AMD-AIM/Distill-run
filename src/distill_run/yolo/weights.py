"""Read official YOLO ``.pt`` checkpoints without importing Ultralytics.

Those files pickle whole ``ultralytics.*`` module objects. The unpickler here
swaps every class outside a small allow-list for an inert placeholder that just
records its state, then walks the placeholders to recover a plain state dict.
No foreign code is executed.
"""

from __future__ import annotations

import logging
import pickle
from collections import OrderedDict
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

logger = logging.getLogger(__name__)

_ALLOWED_PREFIXES = ("torch", "collections", "numpy", "builtins", "copyreg", "_codecs", "datetime")
_BLOCKED = {("builtins", "eval"), ("builtins", "exec"), ("builtins", "getattr"),
            ("builtins", "__import__"), ("builtins", "compile"), ("builtins", "open")}


class Placeholder:
    """Stand-in for a class the unpickler will not import."""

    _qualname = "?"

    def __init__(self, *args, **kwargs):
        self.__dict__["_args"] = args

    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) == 2:
            for part in state:
                if isinstance(part, dict):
                    self.__dict__.update(part)
        elif isinstance(state, dict):
            self.__dict__.update(state)

    def __call__(self, *args, **kwargs):
        return type(self)(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<placeholder {self._qualname}>"


_placeholders: dict[str, type] = {}


def _placeholder(module: str, name: str) -> type:
    key = f"{module}.{name}"
    if key not in _placeholders:
        _placeholders[key] = type(name, (Placeholder,), {"_qualname": key})
    return _placeholders[key]


_PY2_MODULES = {"__builtin__": "builtins", "copy_reg": "copyreg"}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        module = _PY2_MODULES.get(module, module)
        if (module, name) in _BLOCKED:
            raise pickle.UnpicklingError(f"refusing {module}.{name}")
        if module.split(".")[0] in _ALLOWED_PREFIXES:
            return super().find_class(module, name)
        return _placeholder(module, name)


_pickle_module = ModuleType("distill_run_restricted_pickle")
_pickle_module.Unpickler = _Unpickler
_pickle_module.load = lambda f, **kw: _Unpickler(f, **kw).load()
_pickle_module.__dict__.update({k: getattr(pickle, k) for k in ("UnpicklingError", "dumps")})


def _state_dict(obj: Any, prefix: str = "", out: dict | None = None) -> dict:
    out = OrderedDict() if out is None else out
    attrs = obj.__dict__
    for key, value in (attrs.get("_parameters") or {}).items():
        if value is not None:
            out[prefix + key] = value.detach()
    non_persistent = attrs.get("_non_persistent_buffers_set") or set()
    for key, value in (attrs.get("_buffers") or {}).items():
        if value is not None and key not in non_persistent:
            out[prefix + key] = value
    for key, child in (attrs.get("_modules") or {}).items():
        if child is not None:
            _state_dict(child, f"{prefix}{key}.", out)
    return out


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Return ``{"state_dict", "names", "nc", "family", "scale", "stride"}``.

    Accepts official Ultralytics checkpoints and checkpoints written by this
    package (plain dicts that ``torch.load(weights_only=True)`` can read).
    """
    path = Path(path)
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - fall back to the restricted unpickler
        raw = torch.load(
            path, map_location="cpu", weights_only=False, pickle_module=_pickle_module
        )
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        return {**raw, "state_dict": {k: v.float() for k, v in raw["state_dict"].items()}}

    model = (raw.get("ema") or raw.get("model")) if isinstance(raw, dict) else raw
    if model is None or not hasattr(model, "__dict__"):
        raise ValueError(f"{path}: no model found in checkpoint")
    state = {k: v.float() if v.is_floating_point() else v for k, v in _state_dict(model).items()}
    names = getattr(model, "names", None)
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    yaml_cfg = getattr(model, "yaml", None) or {}
    head = _last_layer(model)
    stride = getattr(model, "stride", None)
    return {
        "state_dict": state,
        "names": {int(k): str(v) for k, v in (names or {}).items()},
        "nc": int(getattr(head, "nc", 0) or yaml_cfg.get("nc", 0)),
        "scale": yaml_cfg.get("scale"),
        "yaml_file": yaml_cfg.get("yaml_file"),
        "stride": stride.tolist() if isinstance(stride, torch.Tensor) else None,
    }


def _last_layer(model: Any) -> Any:
    seq = (model.__dict__.get("_modules") or {}).get("model")
    if seq is None:
        return None
    children = list((seq.__dict__.get("_modules") or {}).values())
    return children[-1] if children else None


def apply_state_dict(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Copy matching tensors; class-count dependent layers are skipped on mismatch."""
    own = model.state_dict()
    loaded, mismatched = {}, []
    for key, value in state.items():
        if key in own and own[key].shape == value.shape:
            loaded[key] = value
        elif key in own:
            mismatched.append(key)
    missing = [k for k in own if k not in loaded and k not in mismatched]
    unexpected = [k for k in state if k not in own]
    model.load_state_dict(loaded, strict=False)
    report = {
        "loaded": len(loaded),
        "total": len(own),
        "shape_mismatch": mismatched,
        "missing": missing,
        "unexpected": unexpected,
    }
    logger.info(
        "weights: %d/%d tensors loaded, %d shape mismatches, %d missing, %d unexpected",
        len(loaded), len(own), len(mismatched), len(missing), len(unexpected),
    )
    return report
