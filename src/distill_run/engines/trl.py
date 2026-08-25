"""White-box distillation: teacher and student weights in one process via TRL.

Runs in-process so a Ctrl-C can stop training at a step boundary and still save a
usable adapter.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from .. import preflight
from ..context import RunContext
from ..errors import ConfigError, EngineError, classify
from ..utils import read_jsonl
from .base import Engine, RunResult, config_dataset, make_spec, specs_from_params
from .swift import latest_checkpoint

logger = logging.getLogger(__name__)

PROMPT_KEYS = ("prompt", "instruction", "query", "messages")

DEFAULT_ARGS: dict[str, Any] = {
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-5,
    "bf16": True,
    "gradient_checkpointing": True,
    "logging_steps": 1,
    "save_steps": 100,
    "max_completion_length": 256,
    "temperature": 1.0,
    "beta": 0.5,
    "report_to": "none",
    "dataloader_num_workers": 0,
}

DEFAULT_MODEL_INIT: dict[str, Any] = {
    "torch_dtype": "bfloat16",
    "trust_remote_code": True,
}


class TrlEngine(Engine):
    name = "trl"

    def dataset_specs(self, ctx: RunContext):
        specs = specs_from_params(ctx)
        if "dataset" not in specs:
            from_config = config_dataset(ctx)
            if not from_config:
                raise ConfigError(
                    "no prompt data: pass --data/--dataset (or DATASET_URI), "
                    "or set dataset.path in --config"
                )
            specs["dataset"] = make_spec(ctx, from_config)
        return specs

    def preflight(self, ctx: RunContext) -> None:
        teacher = Path(ctx.require("teacher"))
        student = Path(ctx.require("student"))
        preflight.check_weights(teacher, "teacher")
        preflight.check_weights(student, "student")
        check_model_compatibility(student, teacher)
        if distributed_world_size() > 1:
            if uses_fsdp2(ctx):
                for label, model in (("student", student), ("teacher", teacher)):
                    if has_tied_embeddings(model):
                        raise ConfigError(
                            f"{label} model {model} uses tied input/output embeddings, which "
                            "PyTorch FSDP2 cannot place in separate shard groups; use ZeRO-3 "
                            "or an equivalent checkpoint with tie_word_embeddings=false"
                        )
            for section_name in ("model_init", "teacher_model_init"):
                device_map = ctx.section(section_name).get("device_map")
                if device_map is not None:
                    raise ConfigError(
                        f"{section_name}.device_map is incompatible with distributed training; "
                        "FSDP owns model placement"
                    )
        preflight.check_arg_names(
            config_args(ctx),
            preflight.trl_arg_names(),
            label="TRL DistillationConfig",
            config_key="trl",
        )

    def run(self, ctx: RunContext) -> RunResult:
        from datasets import Dataset
        from transformers import AutoTokenizer
        from trl import DistillationConfig, DistillationTrainer

        class DistributedDistillationTrainer(DistillationTrainer):
            def _generate_single_turn(self, *args, **kwargs):
                # TRL generates before Trainer enters its compute-loss autocast
                # context. Under FSDP this can feed float32 activations into
                # bf16-sharded Qwen3.5 layers.
                with self.accelerator.autocast():
                    return super()._generate_single_turn(*args, **kwargs)

        student = str(ctx.require("student"))
        teacher = str(ctx.require("teacher"))
        output = ctx.output
        output.mkdir(parents=True, exist_ok=True)

        rows = prompt_rows(ctx.dataset("dataset").path)
        logger.info("prompts: %d, student=%s, teacher=%s", len(rows), student, teacher)

        tokenizer = AutoTokenizer.from_pretrained(student, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        final = output / "final"
        try:
            training_args = DistillationConfig(
                output_dir=str(output),
                teacher_model_name_or_path=teacher,
                model_init_kwargs={**DEFAULT_MODEL_INIT, **ctx.section("model_init")},
                teacher_model_init_kwargs={
                    **DEFAULT_MODEL_INIT,
                    **ctx.section("teacher_model_init"),
                },
                **{**DEFAULT_ARGS, **config_args(ctx)},
            )
            trainer = DistributedDistillationTrainer(
                model=student,
                teacher_model=teacher,
                args=training_args,
                train_dataset=Dataset.from_list(rows),
                processing_class=tokenizer,
                peft_config=peft_config(ctx),
                callbacks=[_stop_on_signal(ctx)],
            )
            if trainer.is_fsdp_enabled:
                # DistillationTrainer currently calls Accelerator.prepare_model
                # for the frozen teacher, unlike TRL's DPO/GRPO trainers. That
                # leaves the second model on CPU under FSDP. Use TRL's dedicated
                # helper, which also handles an already-wrapped model.
                from trl.models.utils import prepare_fsdp

                trainer.teacher_model = prepare_fsdp(
                    trainer.teacher_model,
                    trainer.accelerator,
                )
            resume = str(latest_checkpoint(output)) if ctx.resume else None
            if ctx.resume and resume is None:
                logger.warning("--resume requested but no checkpoint found under %s", output)
            trainer.train(resume_from_checkpoint=resume)
            trainer.save_model(str(final))
            trainer.accelerator.wait_for_everyone()
            is_main_process = trainer.is_world_process_zero()
        except Exception as exc:  # noqa: BLE001 - normalise OOM and friends
            logger.exception("TRL training failed")
            raise classify(exc) from exc

        if is_main_process and (not final.is_dir() or not any(final.iterdir())):
            raise EngineError(f"trl finished but {final} is empty")
        return RunResult(
            artifacts=[final] if is_main_process else [],
            message=f"student adapter at {final}" if is_main_process else "worker complete",
        )


def _stop_on_signal(ctx: RunContext):
    """Callback that turns a pending Ctrl-C into a clean stop at a step boundary."""
    from transformers import TrainerCallback

    class StopOnSignal(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001, ARG002
            if ctx.cancellation.requested:
                logger.warning("stopping at step %s and saving", state.global_step)
                control.should_training_stop = True
                control.should_save = True
            return control

    return StopOnSignal()


def prompt_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        candidates = sorted(path.glob("*.jsonl"))
        if not candidates:
            raise ConfigError(f"no .jsonl file found under {path}")
        path = candidates[0]
    prompts: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        if isinstance(row.get("messages"), list):
            prompts.append({"prompt": row["messages"]})
            continue
        text = next((row[k] for k in PROMPT_KEYS if isinstance(row.get(k), str)), None)
        if text is None:
            raise ConfigError(f"{path}: rows need one of {PROMPT_KEYS}; got {sorted(row)}")
        prompts.append({"prompt": [{"role": "user", "content": text}]})
    return prompts


def peft_config(ctx: RunContext):
    section = ctx.section("lora", ctx.section("peft"))
    if not section or section.pop("enabled", True) is False:
        return None
    from peft import LoraConfig

    defaults = {
        "r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    return LoraConfig(**{**defaults, **section})


def config_args(ctx: RunContext) -> dict[str, Any]:
    section = ctx.config.get("trl", ctx.config.get("train", {}))
    if not isinstance(section, dict):
        raise ConfigError("config key 'trl' must be a mapping of DistillationConfig arguments")
    return dict(section)


def distributed_world_size() -> int:
    try:
        return max(1, int(os.environ.get("WORLD_SIZE", "1")))
    except ValueError:
        return 1


def uses_fsdp2(ctx: RunContext) -> bool:
    custom = ctx.param("accelerate_config")
    if not custom:
        return str(ctx.param("distributed_strategy", "zero3")).lower() == "fsdp2"
    try:
        import yaml

        raw = yaml.safe_load(Path(custom).read_text(encoding="utf-8")) or {}
        fsdp = raw.get("fsdp_config", {})
        return (
            str(raw.get("distributed_type", "")).upper() == "FSDP"
            and int(fsdp.get("fsdp_version", 1)) == 2
        )
    except (OSError, TypeError, ValueError):
        return False


def has_tied_embeddings(model: Path) -> bool:
    try:
        raw = json.loads((model / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    text = raw.get("text_config")
    if isinstance(text, dict) and "tie_word_embeddings" in text:
        return bool(text["tie_word_embeddings"])
    return bool(raw.get("tie_word_embeddings", False))


def check_model_compatibility(student: Path, teacher: Path) -> None:
    """Reject incompatible vocabularies before either model is loaded."""
    try:
        from transformers import AutoConfig

        student_config = AutoConfig.from_pretrained(student, trust_remote_code=True)
        teacher_config = AutoConfig.from_pretrained(teacher, trust_remote_code=True)
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read model config: {exc}") from exc

    student_vocab = student_config.get_text_config().vocab_size
    teacher_vocab = teacher_config.get_text_config().vocab_size
    if student_vocab != teacher_vocab:
        raise ConfigError(
            f"teacher/student vocabulary mismatch: teacher={teacher_vocab}, student={student_vocab}"
        )
