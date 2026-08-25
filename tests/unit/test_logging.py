from __future__ import annotations

import logging

import pytest

from distill_run.cli import configure_logging


@pytest.fixture(autouse=True)
def _fresh_logger():
    ours = logging.getLogger("distill_run")
    saved = list(ours.handlers)
    ours.handlers = []
    yield
    ours.handlers = saved


def test_a_framework_reconfiguring_root_logging_does_not_silence_us(capsys):
    # Importing ms-swift calls basicConfig(force=True), which drops handlers from
    # the root logger. Ours must survive that.
    configure_logging()
    logging.basicConfig(force=True)
    logging.getLogger("distill_run.engines.easydistill_swift").info("stage 2/2 starting")
    assert "stage 2/2 starting" in capsys.readouterr().out


def test_log_level_is_configurable(capsys, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    configure_logging()
    logging.getLogger("distill_run.x").info("chatter")
    logging.getLogger("distill_run.x").warning("problem")
    out = capsys.readouterr().out
    assert "chatter" not in out
    assert "problem" in out


def test_non_main_rank_defaults_to_warning(capsys, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.setenv("RANK", "1")
    configure_logging()
    logging.getLogger("distill_run.worker").info("duplicate progress")
    logging.getLogger("distill_run.worker").warning("worker problem")
    out = capsys.readouterr().out
    assert "duplicate progress" not in out
    assert "worker problem" in out
