from __future__ import annotations

from types import SimpleNamespace

import pytest

from distill_run import launcher


def test_single_process_unless_explicitly_requested(monkeypatch: pytest.MonkeyPatch):
    # A container that sees six devices must not silently become a six-process run.
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2,3,4,5,6,7")
    monkeypatch.delenv("NPROC_PER_NODE", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert launcher.visible_device_count() == 6
    assert launcher.world_size() == 1


def test_explicit_request_is_honoured(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2,3,4,5")
    monkeypatch.setenv("NPROC_PER_NODE", "4")
    assert launcher.world_size() == 4


def test_request_is_capped_at_visible_devices(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("NPROC_PER_NODE", "8")
    assert launcher.world_size() == 2


def test_garbage_falls_back_to_one(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NPROC_PER_NODE", "many")
    assert launcher.world_size() == 1


def test_torchrun_wrapping_is_skipped_for_a_single_process():
    argv = ["swift", "sft", "--model", "/models/x"]
    assert launcher.wrap_with_torchrun(argv, 1) == argv
    wrapped = launcher.wrap_with_torchrun(argv, 4)
    assert wrapped[:2] == ["torchrun", "--nproc_per_node=4"]
    assert wrapped[-len(argv) :] == argv


def test_explicit_cli_process_count_takes_precedence(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("NPROC_PER_NODE", "2")
    assert launcher.world_size(4) == 4


@pytest.mark.parametrize(
    ("strategy", "config_name"),
    [
        ("fsdp2", "fsdp2.yaml"),
        ("zero1", "deepspeed_zero1.yaml"),
        ("zero2", "deepspeed_zero2.yaml"),
        ("zero3", "deepspeed_zero3.yaml"),
    ],
)
def test_trl_cli_relaunches_itself_under_accelerate(
    monkeypatch: pytest.MonkeyPatch, strategy: str, config_name: str
):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1,2,3")
    captured = []
    values = {
        "num_processes": 4,
        "distributed_strategy": strategy,
        "main_process_port": 29600,
    }
    ctx = SimpleNamespace(
        engine="trl",
        run_id="generated-id",
        params=SimpleNamespace(origin=lambda _key: None),
        param=lambda key, default=None: values.get(key, default),
    )
    monkeypatch.setattr(launcher.shutil, "which", lambda _name: "/usr/bin/accelerate")
    monkeypatch.setattr(launcher, "stream_command", lambda argv: captured.extend(argv) or 0)

    code = launcher.launch_trl_if_needed(ctx, ["--engine", "trl", "--output", "/tmp/out"])

    assert code == 0
    assert captured[:2] == ["/usr/bin/accelerate", "launch"]
    assert captured[captured.index("--config_file") + 1].endswith(config_name)
    assert "--num_processes" in captured
    assert captured[captured.index("--num_processes") + 1] == "4"
    assert captured[-2:] == ["--run-id", "generated-id"]


def test_trl_worker_does_not_recursively_relaunch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    ctx = SimpleNamespace(
        engine="trl",
        param=lambda key, default=None: 4 if key == "num_processes" else default,
    )
    assert launcher.launch_trl_if_needed(ctx, ["--engine", "trl"]) is None


def test_single_process_training_is_limited_to_one_logical_gpu(monkeypatch: pytest.MonkeyPatch):
    values = {"num_processes": 1, "device": 3}
    ctx = SimpleNamespace(
        engine="swift",
        param=lambda key, default=None: values.get(key, default),
    )
    launcher.configure_single_device(ctx)
    assert launcher.visible_device_count() == 1
    assert launcher.os.environ["HIP_VISIBLE_DEVICES"] == "3"
