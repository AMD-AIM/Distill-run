"""The parameter table.

Single source of truth for the CLI: the argument parser, the per-engine accepted
set, required-field validation and dataset resolution are all derived from
``PARAMS``. Adding a parameter means editing this table, nothing else.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ENGINES: tuple[str, ...] = ("easydistill", "swift", "trl", "easydistill-swift", "noop")

ParamKind = Literal["str", "int", "float", "bool", "path", "dataset", "csv"]

ALL = "*"

DEFAULT_WORK_DIR = os.environ.get("WORK_DIR") or str(Path.home() / ".cache" / "distill-run")


@dataclass(frozen=True)
class Param:
    """One CLI parameter.

    Attributes:
        engines: Engines that accept it, or ``ALL``.
        required: Engines for which a value must exist after merging.
        config_fallback: When true, a missing value is not an error at the
            generic layer because the engine can read it from ``--config``.
        env: Equivalent environment variable; args win over env.
    """

    flag: str
    dest: str
    help: str
    kind: ParamKind = "str"
    engines: tuple[str, ...] | str = ALL
    required: tuple[str, ...] = ()
    config_fallback: bool = False
    env: str | None = None
    default: object | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)
    repeatable: bool = False
    config_paths: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def flags(self) -> tuple[str, ...]:
        return (self.flag, *self.aliases)

    def applies_to(self, engine: str) -> bool:
        return self.engines == ALL or engine in self.engines

    def is_required_for(self, engine: str) -> bool:
        return engine in self.required

    def config_targets(self, engine: str) -> tuple[str, ...]:
        return tuple(path for target_engine, path in self.config_paths if target_engine == engine)


TEACHER_ENGINES = ("easydistill", "easydistill-swift")
TRAIN_ENGINES = ("swift", "trl", "easydistill-swift")
DISTRIBUTED_ENGINES = ("swift", "trl", "easydistill-swift")

PARAMS: tuple[Param, ...] = (
    Param(
        flag="--engine",
        dest="engine",
        help=f"Which engine to run: {' | '.join(ENGINES)}.",
        required=ENGINES,
        env="DISTILL_ENGINE",
    ),
    Param(
        flag="--output",
        dest="output",
        help="Artifact path (JSONL file for easydistill) or directory (everything else).",
        kind="path",
        required=ENGINES,
        env="OUTPUT_PATH",
    ),
    Param(
        flag="--config",
        dest="config",
        help="YAML/JSON file with engine details and hyper-parameters.",
        kind="path",
        env="CONFIG_PATH",
    ),
    Param(
        flag="--config-json",
        dest="config_json",
        help="Inline JSON/YAML engine config; merged over built-in defaults and --config.",
        env="DISTILL_CONFIG_JSON",
    ),
    Param(
        flag="--show-default-config",
        dest="show_default_config",
        help="Print the built-in config for --engine and exit.",
        kind="bool",
        env="DISTILL_SHOW_DEFAULT_CONFIG",
    ),
    Param(
        flag="--models",
        dest="list_models",
        help="Print supported and locally discovered models, then exit.",
        kind="bool",
        env="DISTILL_LIST_MODELS",
    ),
    Param(
        flag="--datasets",
        dest="list_datasets",
        help="Print supported dataset sources, formats and schemas, then exit.",
        kind="bool",
        env="DISTILL_LIST_DATASETS",
    ),
    Param(
        flag="--engines",
        dest="list_engines",
        help="Print supported distillation engines, then exit.",
        kind="bool",
        env="DISTILL_LIST_ENGINES",
    ),
    Param(
        flag="--set",
        dest="config_overrides",
        help="Override any config key as dotted.path=value; may be repeated.",
        repeatable=True,
        env="DISTILL_CONFIG_OVERRIDES",
    ),
    Param(
        flag="--job-type",
        dest="job_type",
        help="EasyDistill job type (default: instruct_distill).",
        engines=TEACHER_ENGINES,
        env="DISTILL_JOB_TYPE",
        config_paths=(
            ("easydistill", "job_type"),
            ("easydistill-swift", "easydistill.job_type"),
        ),
    ),
    Param(
        flag="--backend-type",
        dest="backend_type",
        help="Teacher API backend type (default: openai).",
        engines=TEACHER_ENGINES,
        env="DISTILL_BACKEND_TYPE",
        config_paths=(
            ("easydistill", "backend.type"),
            ("easydistill-swift", "easydistill.backend.type"),
        ),
    ),
    Param(
        flag="--system-prompt",
        dest="system_prompt",
        help='Teacher system prompt (default: "You are a helpful assistant.").',
        engines=TEACHER_ENGINES,
        env="DISTILL_SYSTEM_PROMPT",
        config_paths=(
            ("easydistill", "generation.system_prompt"),
            ("easydistill-swift", "easydistill.generation.system_prompt"),
        ),
    ),
    Param(
        flag="--temperature",
        dest="temperature",
        help="Generation/distillation temperature (default: EasyDistill 0.7; TRL 1.0).",
        kind="float",
        engines=("easydistill", "trl", "easydistill-swift"),
        env="DISTILL_TEMPERATURE",
        config_paths=(
            ("easydistill", "generation.temperature"),
            ("easydistill-swift", "easydistill.generation.temperature"),
            ("trl", "trl.temperature"),
        ),
    ),
    Param(
        flag="--max-tokens",
        dest="max_tokens",
        help="Maximum generated teacher tokens (default: 512; easydistill-swift: 1024).",
        kind="int",
        engines=TEACHER_ENGINES,
        env="DISTILL_MAX_TOKENS",
        config_paths=(
            ("easydistill", "generation.max_tokens"),
            ("easydistill-swift", "easydistill.generation.max_tokens"),
        ),
    ),
    Param(
        flag="--max-workers",
        dest="max_workers",
        help="Concurrent teacher requests (default: 4).",
        kind="int",
        engines=TEACHER_ENGINES,
        env="DISTILL_MAX_WORKERS",
        config_paths=(
            ("easydistill", "generation.max_workers"),
            ("easydistill-swift", "easydistill.generation.max_workers"),
        ),
    ),
    Param(
        flag="--show-progress",
        dest="show_progress",
        help="Show EasyDistill progress (default: true; use --no-show-progress to disable).",
        kind="bool",
        engines=TEACHER_ENGINES,
        env="DISTILL_SHOW_PROGRESS",
        config_paths=(
            ("easydistill", "generation.show_progress"),
            ("easydistill-swift", "easydistill.generation.show_progress"),
        ),
    ),
    Param(
        flag="--instruction-key",
        dest="instruction_key",
        help="Seed JSON instruction field (default: instruction).",
        engines=TEACHER_ENGINES,
        env="DISTILL_INSTRUCTION_KEY",
        config_paths=(
            ("easydistill", "dataset.instruction_key"),
            ("easydistill-swift", "easydistill.dataset.instruction_key"),
        ),
    ),
    Param(
        flag="--skip-empty",
        dest="skip_empty",
        help="Skip empty instructions (default: true; use --no-skip-empty to disable).",
        kind="bool",
        engines=TEACHER_ENGINES,
        env="DISTILL_SKIP_EMPTY",
        config_paths=(
            ("easydistill", "dataset.skip_empty"),
            ("easydistill-swift", "easydistill.dataset.skip_empty"),
        ),
    ),
    Param(
        flag="--min-length",
        dest="min_length",
        help="Minimum instruction length (default: 10).",
        kind="int",
        engines=TEACHER_ENGINES,
        env="DISTILL_MIN_LENGTH",
        config_paths=(
            ("easydistill", "dataset.min_length"),
            ("easydistill-swift", "easydistill.dataset.min_length"),
        ),
    ),
    Param(
        flag="--tuner-type",
        dest="tuner_type",
        help="Swift tuner type (default: lora).",
        engines=("swift", "easydistill-swift"),
        env="DISTILL_TUNER_TYPE",
        config_paths=(
            ("swift", "swift.tuner_type"),
            ("easydistill-swift", "swift.tuner_type"),
        ),
    ),
    Param(
        flag="--lora-rank",
        dest="lora_rank",
        help="LoRA rank (default: 8).",
        kind="int",
        engines=TRAIN_ENGINES,
        env="DISTILL_LORA_RANK",
        config_paths=(
            ("swift", "swift.lora_rank"),
            ("easydistill-swift", "swift.lora_rank"),
            ("trl", "lora.r"),
        ),
    ),
    Param(
        flag="--lora-alpha",
        dest="lora_alpha",
        help="LoRA alpha (default: 32).",
        kind="int",
        engines=TRAIN_ENGINES,
        env="DISTILL_LORA_ALPHA",
        config_paths=(
            ("swift", "swift.lora_alpha"),
            ("easydistill-swift", "swift.lora_alpha"),
            ("trl", "lora.lora_alpha"),
        ),
    ),
    Param(
        flag="--lora-dropout",
        dest="lora_dropout",
        help="TRL LoRA dropout (default: 0.05).",
        kind="float",
        engines=("trl",),
        env="DISTILL_LORA_DROPOUT",
        config_paths=(("trl", "lora.lora_dropout"),),
    ),
    Param(
        flag="--lora-enabled",
        dest="lora_enabled",
        help="Enable TRL LoRA (default: true; use --no-lora-enabled for full tuning).",
        kind="bool",
        engines=("trl",),
        env="DISTILL_LORA_ENABLED",
        config_paths=(("trl", "lora.enabled"),),
    ),
    Param(
        flag="--target-modules",
        dest="target_modules",
        help="Comma-separated TRL LoRA modules (default: q/k/v/o/gate/up/down projections).",
        kind="csv",
        engines=("trl",),
        env="DISTILL_TARGET_MODULES",
        config_paths=(("trl", "lora.target_modules"),),
    ),
    Param(
        flag="--num-train-epochs",
        dest="num_train_epochs",
        help="Training epochs (default: Swift 3; TRL 1).",
        kind="float",
        engines=TRAIN_ENGINES,
        env="DISTILL_NUM_TRAIN_EPOCHS",
        config_paths=(
            ("swift", "swift.num_train_epochs"),
            ("easydistill-swift", "swift.num_train_epochs"),
            ("trl", "trl.num_train_epochs"),
        ),
    ),
    Param(
        flag="--max-steps",
        dest="max_steps",
        help="Maximum optimizer steps (default: TRL 100; Swift framework default).",
        kind="int",
        engines=TRAIN_ENGINES,
        env="DISTILL_MAX_STEPS",
        config_paths=(
            ("swift", "swift.max_steps"),
            ("easydistill-swift", "swift.max_steps"),
            ("trl", "trl.max_steps"),
        ),
    ),
    Param(
        flag="--per-device-train-batch-size",
        dest="per_device_train_batch_size",
        help="Per-device train batch size (default: 1).",
        kind="int",
        engines=TRAIN_ENGINES,
        env="DISTILL_PER_DEVICE_TRAIN_BATCH_SIZE",
        config_paths=(
            ("swift", "swift.per_device_train_batch_size"),
            ("easydistill-swift", "swift.per_device_train_batch_size"),
            ("trl", "trl.per_device_train_batch_size"),
        ),
    ),
    Param(
        flag="--gradient-accumulation-steps",
        dest="gradient_accumulation_steps",
        help="Gradient accumulation steps (default: 8).",
        kind="int",
        engines=TRAIN_ENGINES,
        env="DISTILL_GRADIENT_ACCUMULATION_STEPS",
        config_paths=(
            ("swift", "swift.gradient_accumulation_steps"),
            ("easydistill-swift", "swift.gradient_accumulation_steps"),
            ("trl", "trl.gradient_accumulation_steps"),
        ),
    ),
    Param(
        flag="--learning-rate",
        dest="learning_rate",
        help="Learning rate (default: Swift 1e-4; TRL 1e-5).",
        kind="float",
        engines=TRAIN_ENGINES,
        env="DISTILL_LEARNING_RATE",
        config_paths=(
            ("swift", "swift.learning_rate"),
            ("easydistill-swift", "swift.learning_rate"),
            ("trl", "trl.learning_rate"),
        ),
    ),
    Param(
        flag="--max-length",
        dest="max_length",
        help="Swift maximum sequence length (default: 2048).",
        kind="int",
        engines=("swift", "easydistill-swift"),
        env="DISTILL_MAX_LENGTH",
        config_paths=(
            ("swift", "swift.max_length"),
            ("easydistill-swift", "swift.max_length"),
        ),
    ),
    Param(
        flag="--max-completion-length",
        dest="max_completion_length",
        help="TRL maximum generated completion length (default: 256).",
        kind="int",
        engines=("trl",),
        env="DISTILL_MAX_COMPLETION_LENGTH",
        config_paths=(("trl", "trl.max_completion_length"),),
    ),
    Param(
        flag="--logging-steps",
        dest="logging_steps",
        help="Log interval (default: Swift 5; TRL 1).",
        kind="int",
        engines=TRAIN_ENGINES,
        env="DISTILL_LOGGING_STEPS",
        config_paths=(
            ("swift", "swift.logging_steps"),
            ("easydistill-swift", "swift.logging_steps"),
            ("trl", "trl.logging_steps"),
        ),
    ),
    Param(
        flag="--save-steps",
        dest="save_steps",
        help="Checkpoint interval (default: 200).",
        kind="int",
        engines=TRAIN_ENGINES,
        env="DISTILL_SAVE_STEPS",
        config_paths=(
            ("swift", "swift.save_steps"),
            ("easydistill-swift", "swift.save_steps"),
            ("trl", "trl.save_steps"),
        ),
    ),
    Param(
        flag="--save-total-limit",
        dest="save_total_limit",
        help="Maximum retained checkpoints (default: 2).",
        kind="int",
        engines=TRAIN_ENGINES,
        env="DISTILL_SAVE_TOTAL_LIMIT",
        config_paths=(
            ("swift", "swift.save_total_limit"),
            ("easydistill-swift", "swift.save_total_limit"),
            ("trl", "trl.save_total_limit"),
        ),
    ),
    Param(
        flag="--gradient-checkpointing",
        dest="gradient_checkpointing",
        help="Enable gradient checkpointing (default: true; use --no-gradient-checkpointing).",
        kind="bool",
        engines=TRAIN_ENGINES,
        env="DISTILL_GRADIENT_CHECKPOINTING",
        config_paths=(
            ("swift", "swift.gradient_checkpointing"),
            ("easydistill-swift", "swift.gradient_checkpointing"),
            ("trl", "trl.gradient_checkpointing"),
        ),
    ),
    Param(
        flag="--bf16",
        dest="bf16",
        help="Enable TRL bfloat16 training (default: true; use --no-bf16 to disable).",
        kind="bool",
        engines=("trl",),
        env="DISTILL_BF16",
        config_paths=(("trl", "trl.bf16"),),
    ),
    Param(
        flag="--beta",
        dest="beta",
        help="TRL distillation loss weight (default: 0.5).",
        kind="float",
        engines=("trl",),
        env="DISTILL_BETA",
        config_paths=(("trl", "trl.beta"),),
    ),
    Param(
        flag="--torch-dtype",
        dest="torch_dtype",
        help="Student model dtype (default: bfloat16).",
        engines=TRAIN_ENGINES,
        env="DISTILL_TORCH_DTYPE",
        config_paths=(
            ("swift", "swift.torch_dtype"),
            ("easydistill-swift", "swift.torch_dtype"),
            ("trl", "model_init.torch_dtype"),
        ),
    ),
    Param(
        flag="--teacher-torch-dtype",
        dest="teacher_torch_dtype",
        help="TRL teacher model dtype (default: bfloat16).",
        engines=("trl",),
        env="DISTILL_TEACHER_TORCH_DTYPE",
        config_paths=(("trl", "teacher_model_init.torch_dtype"),),
    ),
    Param(
        flag="--trust-remote-code",
        dest="trust_remote_code",
        help="Trust student model repository code (default: true; use --no-trust-remote-code).",
        kind="bool",
        engines=("trl",),
        env="DISTILL_TRUST_REMOTE_CODE",
        config_paths=(("trl", "model_init.trust_remote_code"),),
    ),
    Param(
        flag="--teacher-trust-remote-code",
        dest="teacher_trust_remote_code",
        help=(
            "Trust teacher model repository code "
            "(default: true; use --no-teacher-trust-remote-code)."
        ),
        kind="bool",
        engines=("trl",),
        env="DISTILL_TEACHER_TRUST_REMOTE_CODE",
        config_paths=(("trl", "teacher_model_init.trust_remote_code"),),
    ),
    Param(
        flag="--run-id",
        dest="run_id",
        help="Run label; defaults to a generated timestamp id.",
        env="RUN_ID",
    ),
    Param(
        flag="--work-dir",
        dest="work_dir",
        help="Scratch directory for downloaded models, datasets and intermediate files.",
        kind="path",
        env="WORK_DIR",
        default=DEFAULT_WORK_DIR,
    ),
    Param(
        flag="--model-cache-dir",
        dest="model_cache_dir",
        help="Where downloaded models are cached; defaults to <work-dir>/models.",
        kind="path",
        engines=TRAIN_ENGINES,
        env="MODEL_CACHE_DIR",
    ),
    Param(
        flag="--model-revision",
        dest="model_revision",
        help="Hub revision or commit used for downloaded teacher and student models.",
        engines=TRAIN_ENGINES,
        env="MODEL_REVISION",
    ),
    Param(
        flag="--resume",
        dest="resume",
        help="Resume from the latest checkpoint already under --output.",
        kind="bool",
        engines=TRAIN_ENGINES,
        env="DISTILL_RESUME",
    ),
    Param(
        flag="--num-processes",
        dest="num_processes",
        help="Local training process/GPU count (default: 1); TRL self-launches Accelerate above 1.",
        kind="int",
        engines=DISTRIBUTED_ENGINES,
        env="DISTILL_NUM_PROCESSES",
        aliases=("--nproc-per-node",),
        default=1,
    ),
    Param(
        flag="--distributed-strategy",
        dest="distributed_strategy",
        help="TRL launcher strategy: fsdp2, zero1, zero2 or zero3 (default: zero3).",
        engines=("trl",),
        env="DISTILL_DISTRIBUTED_STRATEGY",
        default="zero3",
    ),
    Param(
        flag="--accelerate-config",
        dest="accelerate_config",
        help="Custom Accelerate YAML; overrides --distributed-strategy.",
        kind="path",
        engines=("trl",),
        env="ACCELERATE_CONFIG_FILE",
    ),
    Param(
        flag="--main-process-port",
        dest="main_process_port",
        help="Local distributed rendezvous port (default: 29500).",
        kind="int",
        engines=DISTRIBUTED_ENGINES,
        env="MASTER_PORT",
    ),
    Param(
        flag="--device",
        dest="device",
        help="Logical GPU index for a single-process training run (default: 0).",
        kind="int",
        engines=TRAIN_ENGINES,
        env="DISTILL_DEVICE",
        default=0,
    ),
    Param(
        flag="--dataset-cache-dir",
        dest="dataset_cache_dir",
        help="Where downloaded datasets are cached; defaults to <work-dir>/cache.",
        kind="path",
        env="DATASET_CACHE_DIR",
    ),
    Param(
        flag="--dataset-format",
        dest="dataset_format",
        help="Force dataset format: auto, jsonl, json, parquet or arrow (default: auto).",
        env="DATASET_FORMAT",
        default="auto",
    ),
    Param(
        flag="--dataset-revision",
        dest="dataset_revision",
        help="Hub revision or commit for network datasets.",
        env="DATASET_REVISION",
    ),
    Param(
        flag="--dataset-subset",
        dest="dataset_subset",
        help="Hub subset / config name for network datasets.",
        env="DATASET_SUBSET",
    ),
    Param(
        flag="--dataset-split",
        dest="dataset_split",
        help="Dataset split to load (default: train).",
        env="DATASET_SPLIT",
        default="train",
    ),
    Param(
        flag="--dataset-max-samples",
        dest="dataset_max_samples",
        help="Deterministically sample at most this many normalized rows.",
        kind="int",
        env="DATASET_MAX_SAMPLES",
    ),
    Param(
        flag="--dataset-shuffle-seed",
        dest="dataset_shuffle_seed",
        help="Sampling seed used with --dataset-max-samples (default: 42).",
        kind="int",
        env="DATASET_SHUFFLE_SEED",
        default=42,
    ),
    Param(
        flag="--teacher-base-url",
        dest="teacher_base_url",
        help="OpenAI-compatible teacher endpoint, e.g. http://127.0.0.1:8000/v1.",
        engines=TEACHER_ENGINES,
        required=TEACHER_ENGINES,
        config_fallback=True,
        env="TEACHER_BASE_URL",
    ),
    Param(
        flag="--teacher-model-id",
        dest="teacher_model_id",
        help="Model id as served by the teacher endpoint.",
        engines=TEACHER_ENGINES,
        required=TEACHER_ENGINES,
        config_fallback=True,
        env="TEACHER_MODEL_ID",
    ),
    Param(
        flag="--teacher-api-key",
        dest="teacher_api_key",
        help="Teacher API key; vLLM ignores it, remote endpoints need it.",
        engines=TEACHER_ENGINES,
        env="TEACHER_API_KEY",
        default="EMPTY",
    ),
    Param(
        flag="--input",
        dest="input",
        help="Seed instructions: local path or network/Hub URI.",
        kind="dataset",
        engines=TEACHER_ENGINES,
        required=TEACHER_ENGINES,
        config_fallback=True,
        env="INPUT_URI",
    ),
    Param(
        flag="--student",
        dest="student",
        help="Student weights: local path, hf://org/model or ms://org/model.",
        kind="path",
        engines=TRAIN_ENGINES,
        required=TRAIN_ENGINES,
        env="STUDENT_MODEL",
        aliases=("--model",),
    ),
    Param(
        flag="--teacher",
        dest="teacher",
        help="Teacher weights for TRL: local path, hf://org/model or ms://org/model.",
        kind="path",
        engines=("trl",),
        required=("trl",),
        env="TEACHER_MODEL",
    ),
    Param(
        flag="--dataset",
        dest="dataset",
        help="Training data: local path or network/Hub URI (SFT JSONL for swift, prompts for trl).",
        kind="dataset",
        engines=("swift", "trl"),
        required=("swift", "trl"),
        config_fallback=True,
        env="DATASET_URI",
        aliases=("--data",),
    ),
)

_BY_DEST = {p.dest: p for p in PARAMS}


def get(dest: str) -> Param:
    return _BY_DEST[dest]


def params_for(engine: str) -> tuple[Param, ...]:
    return tuple(p for p in PARAMS if p.applies_to(engine))


def dataset_params_for(engine: str) -> tuple[Param, ...]:
    return tuple(p for p in params_for(engine) if p.kind == "dataset")
