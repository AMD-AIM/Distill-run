"""Exit codes and the runtime's single exception type.

Each failure carries a short greppable code for the log line and an exit code a
shell script can branch on.
"""

from __future__ import annotations

from enum import IntEnum


class ExitCode(IntEnum):
    OK = 0
    USAGE = 2
    CONFIG_INVALID = 3
    DATA_INVALID = 4
    DATASET_FETCH_FAILED = 5
    TEACHER_UNAVAILABLE = 6
    PREFLIGHT_FAILED = 7
    ENGINE_FAILED = 8
    OOM = 9
    INTERNAL = 70
    CANCELLED = 130


class DistillRunError(Exception):
    """Base for every expected failure, so the CLI has one exit funnel."""

    code = "EngineFailed"
    exit_code = ExitCode.ENGINE_FAILED

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UsageError(DistillRunError):
    code = "Usage"
    exit_code = ExitCode.USAGE


class ConfigError(DistillRunError):
    code = "ConfigInvalid"
    exit_code = ExitCode.CONFIG_INVALID


class DataError(DistillRunError):
    code = "DataInvalid"
    exit_code = ExitCode.DATA_INVALID


class DatasetFetchError(DistillRunError):
    code = "DatasetFetchFailed"
    exit_code = ExitCode.DATASET_FETCH_FAILED


class TeacherUnavailableError(DistillRunError):
    code = "TeacherDown"
    exit_code = ExitCode.TEACHER_UNAVAILABLE


class PreflightError(DistillRunError):
    code = "PreflightFailed"
    exit_code = ExitCode.PREFLIGHT_FAILED


class EngineError(DistillRunError):
    code = "EngineFailed"
    exit_code = ExitCode.ENGINE_FAILED


class OutOfMemoryError(DistillRunError):
    code = "Oom"
    exit_code = ExitCode.OOM


class CancelledError(DistillRunError):
    code = "Cancelled"
    exit_code = ExitCode.CANCELLED

    def __init__(self, message: str = "cancelled by signal") -> None:
        super().__init__(message)


class InternalError(DistillRunError):
    code = "Internal"
    exit_code = ExitCode.INTERNAL


_OOM_MARKERS = (
    "out of memory",
    "hip out of memory",
    "torch.outofmemoryerror",
)


def classify(exc: BaseException) -> DistillRunError:
    """Map an arbitrary framework exception onto the taxonomy."""
    if isinstance(exc, DistillRunError):
        return exc
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(marker in text for marker in _OOM_MARKERS):
        return OutOfMemoryError(f"{type(exc).__name__}: {exc}")
    return EngineError(f"{type(exc).__name__}: {exc}")
