"""Built-in engine defaults for image-only command execution."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

EASYDISTILL = {
    "job_type": "instruct_distill",
    "backend": {"type": "openai"},
    "generation": {
        "system_prompt": "You are a helpful assistant.",
        "temperature": 0.7,
        "max_tokens": 512,
        "max_workers": 4,
        "show_progress": True,
    },
    "dataset": {
        "instruction_key": "instruction",
        "skip_empty": True,
        "min_length": 10,
    },
}

SWIFT = {
    "swift": {
        "tuner_type": "lora",
        "lora_rank": 8,
        "lora_alpha": 32,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1.0e-4,
        "max_length": 2048,
        "logging_steps": 5,
        "save_steps": 200,
        "save_total_limit": 2,
        "torch_dtype": "bfloat16",
        "gradient_checkpointing": True,
    }
}

TRL = {
    "trl": {
        "num_train_epochs": 1,
        "max_steps": 100,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1.0e-5,
        "bf16": True,
        "gradient_checkpointing": True,
        "logging_steps": 1,
        "save_steps": 200,
        "save_total_limit": 2,
        "max_completion_length": 256,
        "temperature": 1.0,
        "beta": 0.5,
    },
    "lora": {
        "enabled": True,
        "r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    },
    "model_init": {"torch_dtype": "bfloat16", "trust_remote_code": True},
    "teacher_model_init": {"torch_dtype": "bfloat16", "trust_remote_code": True},
}

EASYDISTILL_SWIFT = {
    "easydistill": {
        **EASYDISTILL,
        "generation": {**EASYDISTILL["generation"], "max_tokens": 1024},
    },
    **SWIFT,
}

_BY_ENGINE: dict[str, dict[str, Any]] = {
    "easydistill": EASYDISTILL,
    "swift": SWIFT,
    "trl": TRL,
    "easydistill-swift": EASYDISTILL_SWIFT,
    "noop": {},
}


def engine_defaults(engine: str) -> dict[str, Any]:
    return deepcopy(_BY_ENGINE.get(engine, {}))
