"""Digest checks for downloaded datasets and weights.

A digest is ``sha256:<hex>`` (the Platform's ``content_hash``) or a bare 64-digit
hex string (``base_model.sha256``).
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from .errors import ConfigError, IntegrityError

logger = logging.getLogger(__name__)

_HEX64 = re.compile(r"[0-9a-f]{64}")


def parse_digest(value: str) -> str:
    """Return the lowercase sha256 hex of ``value``; reject other algorithms."""
    text = str(value).strip()
    algo, sep, digest = text.partition(":")
    if not sep:
        algo, digest = "sha256", text
    digest = digest.strip().lower()
    if algo.strip().lower() != "sha256" or not _HEX64.fullmatch(digest):
        raise ConfigError(f"expected a digest like sha256:<64 hex digits>, got {value!r}")
    return digest


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def matches(path: Path, expected: str) -> bool:
    return sha256_file(path) == parse_digest(expected)


def verify(path: Path, expected: str, what: str) -> None:
    """Raise :class:`IntegrityError` (exit 5) unless ``path`` hashes to ``expected``."""
    path = Path(path)
    want = parse_digest(expected)
    if not path.is_file():
        raise IntegrityError(f"{what}: {path} is not a single file, so its sha256 cannot be checked")
    got = sha256_file(path)
    if got != want:
        raise IntegrityError(f"{what}: sha256 mismatch for {path.name}: expected {want}, got {got}")
    logger.info("%s: sha256 ok (%s)", what, want[:12])
