"""Installation diagnostics that do not import heavy frameworks in the core process."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import subprocess
import sys
from dataclasses import asdict, dataclass

from .runtime_env import environments_root, project_root

_ENGINES = {
    "swift": ("swift", "ms-swift", "swift"),
    "trl": ("trl", "trl", "accelerate"),
    "easydistill": ("easydistill", "easydistill", None),
}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "missing"


def _engine_check(name: str, module: str, package: str, executable: str | None) -> Check:
    python = environments_root() / name / "bin" / "python"
    if not python.is_file():
        return Check(name, False, f"environment missing: {python}")
    if executable and not (python.parent / executable).is_file():
        return Check(name, False, f"executable missing: {python.parent / executable}")
    code = (
        "import importlib.metadata as m, importlib.util as u;"
        f"assert u.find_spec({module!r}) is not None;"
        f"print(m.version({package!r}))"
    )
    try:
        result = subprocess.run(  # noqa: S603
            [str(python), "-c", code],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return Check(name, False, str(exc))
    return Check(name, True, f"{python} ({result.stdout.strip()})")


def collect(selected: list[str] | None = None) -> list[Check]:
    root = project_root()
    checks = [
        Check("core", True, f"distill-run {_package_version('distill-run')} ({sys.executable})"),
        Check("project", (root / "pyproject.toml").is_file(), str(root)),
    ]
    for name in selected or list(_ENGINES):
        module, package, executable = _ENGINES[name]
        checks.append(_engine_check(name, module, package, executable))

    for name in ("trl", "easydistill"):
        path = root / "third_party" / name
        checks.append(Check(f"submodule:{name}", path.is_dir() and any(path.iterdir()), str(path)))
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="distill-run doctor")
    parser.add_argument("--engine", action="append", choices=tuple(_ENGINES))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    checks = collect(args.engine)
    if args.json:
        print(json.dumps([asdict(check) for check in checks], indent=2))
    else:
        for check in checks:
            marker = "OK" if check.ok else "FAIL"
            print(f"[{marker:4s}] {check.name}: {check.detail}")
    return 0 if all(check.ok for check in checks) else 1
