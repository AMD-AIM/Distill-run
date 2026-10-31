"""scripts/setup_envs.py: engine venvs on top of a venv-based image."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import venv
from importlib import metadata
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def setup_envs():
    spec = importlib.util.spec_from_file_location("setup_envs", REPO / "scripts" / "setup_envs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _purelib(python: Path) -> str:
    return subprocess.check_output(
        [str(python), "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
        text=True,
    ).strip()


@pytest.fixture
def parent(tmp_path: Path) -> Path:
    path = tmp_path / "parent"
    venv.EnvBuilder(with_pip=False).create(path)
    return path


def test_base_site_packages_returns_the_purelib_of_a_parent_venv(setup_envs, parent):
    python = parent / "bin" / "python"
    assert setup_envs.base_site_packages(str(python)) == Path(_purelib(python))


def test_base_site_packages_is_none_for_a_plain_interpreter(setup_envs):
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    plain = Path(sys.base_prefix) / "bin" / version
    assert setup_envs.base_site_packages(str(plain)) is None


def test_create_environment_sees_parent_venv_packages(setup_envs, parent, tmp_path):
    parent_python = parent / "bin" / "python"
    parent_lib = Path(_purelib(parent_python))
    (parent_lib / "r156i_probe.py").write_text("VALUE = 1\n", encoding="utf-8")

    child = tmp_path / "child"
    python = setup_envs.create_environment(child, str(parent_python))

    pth = Path(_purelib(python)) / setup_envs.BASE_PTH_NAME
    assert pth.read_text(encoding="utf-8").strip() == str(parent_lib)
    assert subprocess.run([str(python), "-c", "import r156i_probe"]).returncode == 0
    assert (child / ".distill-run-runtime").is_file()


def test_base_constraints_pins_the_torch_family(setup_envs, tmp_path, monkeypatch):
    versions = {
        "torch": "2.10.0+rocm7.2.4.lw.git3d3aa833",
        "torchvision": "0.25.0+rocm7.2.4.git82df5f59",
        "torchaudio": "2.10.0+rocm7.2.4.git5047768f",
        "triton": "3.6.0+rocm7.2.4.git4ed88892",
    }

    def version(name: str) -> str:
        if name not in versions:
            raise metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(setup_envs, "STATE", tmp_path)
    monkeypatch.setattr(setup_envs.metadata, "version", version)
    path = setup_envs.base_constraints()
    assert path.read_text(encoding="utf-8").splitlines() == [
        f"{name}=={value}" for name, value in versions.items()
    ]
