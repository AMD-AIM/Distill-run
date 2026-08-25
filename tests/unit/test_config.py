from __future__ import annotations

import pytest

from distill_run.config import (
    apply_config_overrides,
    check_required,
    deep_merge,
    expand_env,
    expansion_env,
    load_config,
    load_inline_config,
    merge_sources,
)
from distill_run.defaults import engine_defaults
from distill_run.errors import ConfigError, UsageError


def test_args_win_over_env():
    resolved = merge_sources(
        "easydistill",
        {"engine": "easydistill", "teacher_base_url": "http://from-arg/v1"},
        {"TEACHER_BASE_URL": "http://from-env/v1"},
    )
    assert resolved.get("teacher_base_url") == "http://from-arg/v1"
    assert resolved.origin("teacher_base_url") == "arg"


def test_env_fills_in_when_arg_absent():
    resolved = merge_sources(
        "easydistill", {"engine": "easydistill"}, {"TEACHER_BASE_URL": "http://from-env/v1"}
    )
    assert resolved.get("teacher_base_url") == "http://from-env/v1"
    assert resolved.origin("teacher_base_url") == "env"


def test_table_default_is_last_resort():
    resolved = merge_sources("easydistill", {"engine": "easydistill"}, {})
    assert resolved.get("teacher_api_key") == "EMPTY"
    assert resolved.origin("teacher_api_key") == "default"


def test_engine_rejects_foreign_params():
    with pytest.raises(UsageError, match="--teacher-base-url"):
        merge_sources("trl", {"engine": "trl", "teacher_base_url": "http://x/v1"}, {})


def test_hard_requirement_fails_fast():
    with pytest.raises(UsageError, match="--output"):
        check_required(merge_sources("trl", {"engine": "trl"}, {}))


def test_config_fallback_requirements_are_deferred_to_the_engine():
    resolved = merge_sources(
        "trl",
        {"engine": "trl", "output": "/tmp/out", "student": "/m/s", "teacher": "/m/t"},
        {},
    )
    assert [p.dest for p in check_required(resolved)] == ["dataset"]


def test_easydistill_swift_needs_both_a_teacher_url_and_a_student():
    with pytest.raises(UsageError, match="--student"):
        check_required(
            merge_sources(
                "easydistill-swift",
                {"engine": "easydistill-swift", "output": "/tmp/out"},
                {"TEACHER_BASE_URL": "http://t/v1", "TEACHER_MODEL_ID": "m"},
            )
        )


def test_easydistill_swift_does_not_accept_a_teacher_weights_path():
    with pytest.raises(UsageError, match="--teacher"):
        merge_sources(
            "easydistill-swift",
            {"engine": "easydistill-swift", "teacher": "/models/t"},
            {},
        )


def test_env_expansion_supports_defaults():
    assert expand_env("${A:-fallback}", {}) == "fallback"
    assert expand_env("${A:-fallback}", {"A": "set"}) == "set"


def test_env_expansion_rejects_undefined():
    with pytest.raises(ConfigError, match="MISSING"):
        expand_env("${MISSING}", {})


def test_cli_values_are_visible_to_config_expansion(tmp_path):
    # A config written as ${TEACHER_BASE_URL} must resolve even when the value
    # arrived as --teacher-base-url and the env var was never set.
    resolved = merge_sources(
        "easydistill", {"engine": "easydistill", "teacher_base_url": "http://from-arg/v1"}, {}
    )
    path = tmp_path / "c.yaml"
    path.write_text("backend:\n  base_url: ${TEACHER_BASE_URL}\n", encoding="utf-8")
    cfg = load_config(path, expansion_env(resolved, {}))
    assert cfg["backend"]["base_url"] == "http://from-arg/v1"


def test_load_config_missing_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml", {})


def test_load_config_rejects_a_non_mapping(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path, {})


def test_inline_config_and_dotted_overrides_are_typed():
    config = deep_merge(
        engine_defaults("trl"),
        load_inline_config('{"trl": {"learning_rate": 0.00002}}'),
    )
    config = apply_config_overrides(
        config,
        ["trl.max_steps=12", "trl.bf16=false", "lora.target_modules=[q_proj, v_proj]"],
    )
    assert config["trl"]["learning_rate"] == 0.00002
    assert config["trl"]["max_steps"] == 12
    assert config["trl"]["bf16"] is False
    assert config["lora"]["target_modules"] == ["q_proj", "v_proj"]


def test_engine_defaults_are_independent_copies():
    first = engine_defaults("swift")
    first["swift"]["learning_rate"] = 9
    assert engine_defaults("swift")["swift"]["learning_rate"] == 1.0e-4


def test_set_requires_dotted_key_value_syntax():
    with pytest.raises(UsageError, match="dotted.path=value"):
        apply_config_overrides({}, ["broken"])
