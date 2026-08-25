"""Config loading and value precedence: CLI args > environment > config file.

Engine config files are not merged into the parameter table. A parameter marked
``config_fallback`` may be absent from both args and env because the engine reads
it from its own config structure, so the required check defers to the engine.
"""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import params as P
from .errors import ConfigError, UsageError

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

Source = str  # "arg" | "env" | "default"


# --- config files ------------------------------------------------------------


def expand_env(value: Any, env: dict[str, str] | None = None) -> Any:
    """Recursively expand ``${VAR}`` and ``${VAR:-default}`` in a config tree."""
    source = env if env is not None else dict(os.environ)

    def sub(match: re.Match[str]) -> str:
        name, fallback = match.group(1), match.group(2)
        if name in source:
            return source[name]
        if fallback is not None:
            return fallback
        raise ConfigError(f"config references undefined environment variable ${{{name}}}")

    if isinstance(value, str):
        return _ENV_PATTERN.sub(sub, value)
    if isinstance(value, dict):
        return {k: expand_env(v, source) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v, source) for v in value]
    return value


def load_config(path: str | Path | None, env: dict[str, str] | None = None) -> dict[str, Any]:
    if path is None:
        return {}
    file = Path(path)
    if not file.is_file():
        raise ConfigError(f"config file not found: {file}")
    text = file.read_text(encoding="utf-8")
    try:
        raw = json.loads(text) if file.suffix == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot parse config {file}: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config {file} must be a mapping at the top level")
    return expand_env(raw, env)


def dump_yaml(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def load_inline_config(text: str | None, env: dict[str, str] | None = None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"cannot parse --config-json: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("--config-json must contain an object/mapping")
    return expand_env(raw, env)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def apply_config_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    result = deepcopy(config)
    for expression in overrides or []:
        if "=" not in expression:
            raise UsageError(f"--set expects dotted.path=value, got {expression!r}")
        dotted, raw = expression.split("=", 1)
        keys = [key.strip() for key in dotted.split(".") if key.strip()]
        if not keys:
            raise UsageError(f"--set has an empty key: {expression!r}")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise UsageError(f"cannot parse --set {expression!r}: {exc}") from exc
        if isinstance(value, str) and re.fullmatch(
            r"[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?", value
        ):
            value = float(value) if "." in value or "e" in value.lower() else int(value)
        cursor = result
        for key in keys[:-1]:
            existing = cursor.setdefault(key, {})
            if not isinstance(existing, dict):
                raise UsageError(f"--set cannot descend through non-mapping key {key!r}")
            cursor = existing
        cursor[keys[-1]] = value
    return result


def apply_param_config_overrides(config: dict[str, Any], resolved: Resolved) -> dict[str, Any]:
    """Apply explicit vLLM-style flags after generic config overrides."""
    result = deepcopy(config)
    for param in P.params_for(resolved.engine):
        if resolved.origin(param.dest) not in {"arg", "env"}:
            continue
        for dotted in param.config_targets(resolved.engine):
            cursor = result
            keys = dotted.split(".")
            for key in keys[:-1]:
                existing = cursor.setdefault(key, {})
                if not isinstance(existing, dict):
                    raise UsageError(f"{param.flag} cannot override non-mapping key {key!r}")
                cursor = existing
            cursor[keys[-1]] = deepcopy(resolved.get(param.dest))
    return result


# --- parameter merging -------------------------------------------------------


@dataclass
class Resolved:
    """Merged parameter values plus where each one came from."""

    engine: str
    values: dict[str, Any] = field(default_factory=dict)
    origins: dict[str, Source] = field(default_factory=dict)

    def get(self, dest: str, default: Any = None) -> Any:
        value = self.values.get(dest)
        return default if value is None else value

    def origin(self, dest: str) -> Source | None:
        return self.origins.get(dest)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.values.items() if v is not None}


def _coerce(param: P.Param, raw: Any) -> Any:
    if raw is None:
        return None
    if param.repeatable:
        if isinstance(raw, list):
            return raw
        try:
            parsed = json.loads(str(raw))
        except json.JSONDecodeError:
            return [str(raw)]
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise UsageError(f"{param.env} expects a JSON array of dotted.path=value strings")
        return parsed
    if param.kind == "bool":
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}
    if param.kind == "int":
        try:
            return int(raw)
        except ValueError as exc:
            raise UsageError(f"{param.flag} expects an integer, got {raw!r}") from exc
    if param.kind == "float":
        try:
            return float(raw)
        except ValueError as exc:
            raise UsageError(f"{param.flag} expects a number, got {raw!r}") from exc
    if param.kind == "csv":
        if isinstance(raw, list):
            return raw
        return [item.strip() for item in str(raw).split(",") if item.strip()]
    return raw


def merge_sources(engine: str, args: dict[str, Any], env: dict[str, str]) -> Resolved:
    resolved = Resolved(engine=engine)
    for param in P.params_for(engine):
        raw, origin = None, None
        if args.get(param.dest) is not None:
            raw, origin = args[param.dest], "arg"
        elif param.env and env.get(param.env):
            raw, origin = env[param.env], "env"
        elif param.default is not None:
            raw, origin = param.default, "default"
        resolved.values[param.dest] = _coerce(param, raw)
        if origin is not None:
            resolved.origins[param.dest] = origin

    known = {p.dest for p in P.PARAMS}
    foreign = [
        dest
        for dest, value in args.items()
        if value is not None and dest not in resolved.values and dest in known
    ]
    if foreign:
        flags = ", ".join(P.get(dest).flag for dest in sorted(foreign))
        raise UsageError(f"engine '{engine}' does not accept: {flags}")
    return resolved


def expansion_env(resolved: Resolved, env: dict[str, str]) -> dict[str, str]:
    """Environment used to expand ``${VAR}`` inside a config file.

    A config may say ``${TEACHER_BASE_URL}`` while the value arrived as
    ``--teacher-base-url``. Seeding the expansion environment with the merged
    values makes both styles work, with args still winning.
    """
    seeded = dict(env)
    for param in P.params_for(resolved.engine):
        value = resolved.values.get(param.dest)
        if param.env and value is not None:
            seeded[param.env] = str(value)
    return seeded


def check_required(resolved: Resolved) -> list[P.Param]:
    """Fail on missing hard requirements; return the ones the engine must verify."""
    missing: list[str] = []
    deferred: list[P.Param] = []
    for param in P.params_for(resolved.engine):
        if not param.is_required_for(resolved.engine):
            continue
        if resolved.values.get(param.dest) is not None:
            continue
        if param.config_fallback:
            deferred.append(param)
        else:
            hint = f" (or {param.env})" if param.env else ""
            missing.append(f"{param.flag}{hint}")
    if missing:
        raise UsageError(f"engine '{resolved.engine}' requires: {', '.join(sorted(missing))}")
    return deferred
