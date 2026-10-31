"""Keep registered credentials out of log lines, whichever logger emits them.

Framework errors can quote a credential back (``Illegal header value b'Bearer …'``),
so the filter goes on handlers rather than relying on every call site to be careful.
"""

from __future__ import annotations

import logging

MASK = "***"
_SECRETS: set[str] = set()


def register(value: str) -> None:
    if len(value) >= 4:
        _SECRETS.add(value)


def scrub(text: str) -> str:
    for secret in sorted(_SECRETS, key=len, reverse=True):
        text = text.replace(secret, MASK)
    return text


class SecretFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not _SECRETS:
            return True
        record.msg = scrub(record.getMessage())
        record.args = None
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = scrub(record.exc_text)
        if record.stack_info:
            record.stack_info = scrub(record.stack_info)
        return True


def protect(*loggers: logging.Logger) -> None:
    """Add the filter to every handler of ``loggers`` (and the root logger)."""
    for logger in (*loggers, logging.getLogger()):
        for handler in logger.handlers:
            if not any(isinstance(f, SecretFilter) for f in handler.filters):
                handler.addFilter(SecretFilter())
