"""The parameter table encodes which engine may see what; guard those rules."""

from __future__ import annotations

import pytest

from distill_run import params as P
from distill_run.cli import build_parser


def test_parser_exposes_every_flag_and_alias():
    parser = build_parser()
    known = {opt for action in parser._actions for opt in action.option_strings}
    for param in P.PARAMS:
        for flag in param.flags:
            assert flag in known, f"{flag} is not wired into the parser"


def test_every_param_has_an_env_equivalent():
    # env.sh injects most values through the environment, so a flag with no env
    # var would be unreachable from run.sh.
    assert [p.flag for p in P.PARAMS if not p.env] == []


@pytest.mark.parametrize("engine", P.ENGINES)
def test_required_params_are_accepted_by_their_engine(engine: str):
    for param in P.PARAMS:
        if param.is_required_for(engine):
            assert param.applies_to(engine), f"{param.flag} required by {engine} but not accepted"


def test_generation_never_sees_weights():
    easydistill = {p.dest for p in P.params_for("easydistill")}
    assert "student" not in easydistill
    assert "teacher" not in easydistill


def test_training_engines_never_see_a_teacher_url():
    for engine in ("swift", "trl"):
        params = {p.dest for p in P.params_for(engine)}
        assert "teacher_base_url" not in params
        assert "teacher_model_id" not in params


def test_white_box_uses_weights_and_easydistill_swift_uses_a_url():
    assert "teacher" in {p.dest for p in P.params_for("trl")}
    assert "teacher" not in {p.dest for p in P.params_for("easydistill-swift")}
    assert "teacher_base_url" in {p.dest for p in P.params_for("easydistill-swift")}


def test_dataset_params_are_the_ones_that_get_resolved():
    assert {p.dest for p in P.dataset_params_for("easydistill")} == {"input"}
    assert {p.dest for p in P.dataset_params_for("swift")} == {"dataset"}
    assert {p.dest for p in P.dataset_params_for("easydistill-swift")} == {"input"}
