"""Human-readable capability catalogs for CLI discovery and error hints."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from .runtime_env import project_root

ENGINE_CATALOG: dict[str, dict[str, str]] = {
    "easydistill": {
        "framework": "EasyDistill",
        "mode": "black-box generation",
        "input": "seed prompts + OpenAI-compatible teacher API",
    },
    "swift": {
        "framework": "ms-swift",
        "mode": "student SFT",
        "input": "SFT dataset + local or Hub student weights",
    },
    "trl": {
        "framework": "TRL DistillationTrainer",
        "mode": "white-box logits distillation",
        "input": "prompt dataset + local or Hub teacher and student weights",
    },
    "easydistill-swift": {
        "framework": "EasyDistill + ms-swift",
        "mode": "black-box generation followed by student SFT",
        "input": "seed prompts + teacher API + local or Hub student weights",
    },
    "noop": {
        "framework": "built-in",
        "mode": "CLI test engine",
        "input": "none",
    },
}


def supported_engine_names(*, include_internal: bool = True) -> tuple[str, ...]:
    names = tuple(ENGINE_CATALOG)
    return names if include_internal else tuple(name for name in names if name != "noop")


def _model_roots() -> list[Path]:
    configured = [
        os.environ.get("MODELS_DIR"),
        os.environ.get("MODEL_ROOT"),
        str(Path(os.environ.get("WORK_DIR", "")) / "downloads")
        if os.environ.get("WORK_DIR")
        else None,
    ]
    candidates = [
        *(Path(value) for value in configured if value),
        Path("/models"),
        project_root() / "work" / "downloads",
    ]
    result: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def _model_info(path: Path) -> dict[str, Any] | None:
    config = path / "config.json"
    if not config.is_file():
        return None
    try:
        raw = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"path": str(path), "status": "invalid config.json"}
    text = raw.get("text_config") if isinstance(raw.get("text_config"), dict) else raw
    return {
        "path": str(path),
        "model_type": text.get("model_type") or raw.get("model_type") or "unknown",
        "vocab_size": text.get("vocab_size") or raw.get("vocab_size") or "unknown",
        "tied_embeddings": bool(
            text.get("tie_word_embeddings", raw.get("tie_word_embeddings", False))
        ),
    }


def model_catalog() -> dict[str, Any]:
    discovered = []
    roots = _model_roots()
    for root in roots:
        if not root.is_dir():
            continue
        own = _model_info(root)
        if own:
            discovered.append(own)
        try:
            children = sorted(path for path in root.iterdir() if path.is_dir())
        except OSError:
            continue
        discovered.extend(info for path in children if (info := _model_info(path)))
    return {
        "models": {
            "supported": [
                "local Hugging Face Transformers causal-language-model checkpoints",
                "Hugging Face Hub models via hf://org/model",
                "ModelScope models via ms://org/model",
                "OpenAI-compatible teacher model IDs for EasyDistill",
            ],
            "validated_families": ["Qwen3", "Qwen3.5"],
            "constraints": [
                "TRL teacher and student vocabularies must match",
                "FSDP2 requires untied input/output embeddings; ZeRO-3 supports tied models",
            ],
            "search_roots": [str(path) for path in roots],
            "discovered": discovered,
        }
    }


def dataset_catalog() -> dict[str, Any]:
    examples = sorted(
        str(path) for path in (project_root() / "examples").glob("*") if path.is_file()
    )
    return {
        "datasets": {
            "sources": [
                "local path or file://",
                "http:// or https://",
                "hf://org/dataset",
                "ms://org/dataset",
            ],
            "formats": ["JSONL", "JSON", "Parquet", "Arrow"],
            "accepted_prompt_fields": ["instruction", "prompt", "question", "query", "problem"],
            "accepted_response_fields": [
                "output",
                "response",
                "answer",
                "completion",
                "solution",
            ],
            "chat_fields": ["messages", "conversations"],
            "bundled_examples": examples,
        }
    }


def engine_catalog() -> dict[str, Any]:
    return {"engines": ENGINE_CATALOG}


def model_error_hint() -> str:
    discovered = model_catalog()["models"]["discovered"]
    paths = [item["path"] for item in discovered if item.get("status") is None]
    if paths:
        return "\navailable local models:\n" + "\n".join(f"  {path}" for path in paths)
    return "\nrun `distill-run --models` to see supported model types and search roots"


def dataset_error_hint() -> str:
    examples = dataset_catalog()["datasets"]["bundled_examples"]
    available = "\navailable bundled datasets:\n" + "\n".join(f"  {path}" for path in examples)
    return available + "\nrun `distill-run --datasets` for supported sources and formats"


def print_catalog(*, models: bool, datasets: bool, engines: bool) -> None:
    payload: dict[str, Any] = {}
    if models:
        payload.update(model_catalog())
    if datasets:
        payload.update(dataset_catalog())
    if engines:
        payload.update(engine_catalog())
    print(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False).rstrip())
