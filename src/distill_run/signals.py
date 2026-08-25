"""Ctrl-C / SIGTERM handling: record the request, stop at a safe point.

Killing a trainer mid-step leaves a half-written adapter, so engines poll
:attr:`Cancellation.requested` and save before returning. A second signal exits
immediately.
"""

from __future__ import annotations

import logging
import signal
import threading
import time

logger = logging.getLogger(__name__)


class Cancellation:
    def __init__(self) -> None:
        self._event = threading.Event()
        self.signal_number: int | None = None
        self.requested_at: float | None = None

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    def request(self, signal_number: int | None = None) -> None:
        self.signal_number = signal_number
        self.requested_at = time.time()
        self._event.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)


def install_handlers(signals: tuple[int, ...] = (signal.SIGTERM, signal.SIGINT)) -> Cancellation:
    cancellation = Cancellation()

    def handler(signum: int, _frame: object) -> None:
        if cancellation.requested:
            logger.warning("received signal %s again; exiting now", signum)
            raise KeyboardInterrupt
        logger.warning("received signal %s; will stop at the next safe point", signum)
        cancellation.request(signum)

    for sig in signals:
        try:
            signal.signal(sig, handler)
        except ValueError:
            logger.debug("cannot install handler for signal %s (not main thread)", sig)
    return cancellation
