#!/usr/bin/env python3
"""Create reproducible per-engine environments with standard venv and pip."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
import venv
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENVS = ROOT / ".venvs"
STATE = ROOT / ".distill"
SOURCES = {
    "trl": ROOT / "third_party" / "trl",
    "easydistill": ROOT / "third_party" / "easydistill",
}
SUPPORTED_ENGINES = ("swift", "trl", "easydistill")


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)  # noqa: S603


def initialise_submodules() -> None:
    if not (ROOT / ".git").exists():
        return
    run(["git", "submodule", "sync", "--recursive"])
    # The pinned projects have long histories; a fresh install only needs the
    # recorded commits. The mirror can fetch an older pinned commit separately
    # when it is not at the current shallow tip.
    run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"])


def create_environment(path: Path, base_python: str) -> Path:
    python = path / "bin" / "python"
    if not python.is_file():
        if Path(base_python).resolve() != Path(sys.executable).resolve():
            run(
                [
                    base_python,
                    "-m",
                    "venv",
                    "--system-site-packages",
                    str(path),
                ]
            )
        else:
            venv.EnvBuilder(with_pip=True, system_site_packages=True).create(path)
    (path / ".distill-run-runtime").write_text(str(path.resolve()) + "\n", encoding="utf-8")
    return python


def base_constraints() -> Path:
    STATE.mkdir(exist_ok=True)
    path = STATE / "base-constraints.txt"
    packages = []
    for name in ("torch", "vllm"):
        try:
            packages.append(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            pass
    path.write_text("\n".join(packages) + ("\n" if packages else ""), encoding="utf-8")
    return path


def pip_args(args: argparse.Namespace, constraints: Path) -> list[str]:
    result = ["--constraint", str(constraints)]
    if args.offline:
        result += ["--no-index", "--find-links", str(ROOT / "wheelhouse")]
    elif args.index_url:
        result += ["--index-url", args.index_url]
    return result


def pip_install(
    python: Path,
    values: list[str],
    args: argparse.Namespace,
    constraints: Path,
    *,
    no_deps: bool = False,
    ignore_installed: bool = False,
) -> None:
    command = [str(python), "-m", "pip", "install", "--no-build-isolation"]
    command += pip_args(args, constraints)
    if no_deps:
        command.append("--no-deps")
    if ignore_installed:
        command.append("--ignore-installed")
    command += values
    run(command)


def install_core(
    python: Path, args: argparse.Namespace, constraints: Path, *, development: bool
) -> None:
    target = ".[dev,dataset,hub]" if development else "."
    pip_install(python, ["-e", target], args, constraints)


def install_engine(name: str, python: Path, args: argparse.Namespace, constraints: Path) -> None:
    # Install the shared CLI in every worker environment, but keep its dependency
    # graph owned by that environment's explicit requirements file.
    pip_install(python, ["-e", "."], args, constraints, no_deps=True)
    requirement = ROOT / "requirements" / f"{args.profile}-{name}.txt"
    if requirement.is_file():
        pip_install(python, ["-r", str(requirement)], args, constraints)

    if name == "swift":
        # ms-swift currently declares a TRL range incompatible with our custom
        # distillation fork. It is isolated here and its runtime set is explicit.
        pip_install(
            python,
            ["ms-swift==4.5.2"],
            args,
            constraints,
            no_deps=True,
            ignore_installed=True,
        )
    elif name == "trl":
        # A system-site package does not place its console script in the venv.
        # Force only the small launcher package locally so all ranks use this
        # environment's Python and editable TRL fork.
        pip_install(
            python,
            ["accelerate>=1.14"],
            args,
            constraints,
            no_deps=True,
            ignore_installed=True,
        )
        source = SOURCES[name]
        if not (source / "pyproject.toml").is_file():
            raise SystemExit(f"{name} submodule is missing: {source}")
        pip_install(python, ["-e", str(source)], args, constraints, no_deps=True)
    else:
        source = SOURCES[name]
        if not (source / "pyproject.toml").is_file():
            raise SystemExit(f"{name} submodule is missing: {source}")
        pip_install(python, ["-e", str(source)], args, constraints)


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def write_lock(profile: str, engines: list[str], base_python: str, core_dir: Path) -> None:
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "profile": profile,
        "base_python": base_python,
        "core_environment": str(core_dir),
        "engines": engines,
        "submodules": {name: git_revision(path) for name, path in SOURCES.items()},
    }
    (STATE / "environment.lock.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="rocm714")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--core-dir", default=".venv")
    parser.add_argument("--engines", default=",".join(SUPPORTED_ENGINES))
    parser.add_argument("--index-url", default=os.environ.get("PIP_INDEX_URL"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--skip-submodules", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    engines = [name.strip() for name in args.engines.split(",") if name.strip()]
    unknown = sorted(set(engines) - set(SUPPORTED_ENGINES))
    if unknown:
        raise SystemExit(f"unknown engines: {', '.join(unknown)}")
    if not args.skip_submodules:
        initialise_submodules()
    if args.check_only:
        for name in engines:
            print(f"{name}: {'ready' if (ENVS / name / 'bin/python').is_file() else 'missing'}")
        return 0

    constraints = base_constraints()
    core_dir = Path(args.core_dir)
    if not core_dir.is_absolute():
        core_dir = ROOT / core_dir
    core_python = create_environment(core_dir, args.python)
    install_core(core_python, args, constraints, development=True)
    for name in engines:
        python = create_environment(ENVS / name, args.python)
        install_engine(name, python, args, constraints)
    write_lock(args.profile, engines, args.python, core_dir)
    doctor = [str(core_python), "-m", "distill_run", "doctor"]
    for name in engines:
        doctor += ["--engine", name]
    run(doctor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
