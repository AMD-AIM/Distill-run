"""``distill-run``: one entrypoint, one run, one exit code.

Flow: parse -> merge (args > env) -> load config -> cheap preflight -> resolve
datasets to local paths -> run engine.

Everything before the engine is CPU-only and takes seconds, so a misconfigured
run fails before any weights are loaded.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from . import launcher, preflight
from . import params as P
from .config import (
    apply_config_overrides,
    apply_param_config_overrides,
    check_required,
    deep_merge,
    expansion_env,
    load_config,
    load_inline_config,
    merge_sources,
)
from .context import RunContext, generate_run_id, prepare_environment
from .dataset import resolve
from .defaults import engine_defaults
from .engines import registry
from .engines.base import Engine, RunResult
from .errors import CancelledError, DistillRunError, ExitCode, UsageError, classify
from .runtime_env import maybe_run_in_engine_environment
from .signals import install_handlers

logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_LOG_DATEFMT = "%H:%M:%S"


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def configure_logging(level: str | None = None) -> None:
    """Log through our own logger, line-buffered, ignoring the root logger.

    The handler goes on the ``distill_run`` logger with propagation off, because
    importing a framework can reconfigure root logging — ms-swift does, and it
    used to silence every line we logged after the first framework import.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(line_buffering=True)
    ours = logging.getLogger("distill_run")
    configured = level or os.environ.get("LOG_LEVEL")
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    ours.setLevel((configured or ("INFO" if rank == 0 else "WARNING")).upper())
    ours.propagate = False
    if not any(getattr(h, "_distill_run", False) for h in ours.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
        handler._distill_run = True  # type: ignore[attr-defined]
        ours.handlers = [handler]


def build_parser() -> argparse.ArgumentParser:
    """Derive the parser from the parameter table so the two cannot drift."""
    parser = argparse.ArgumentParser(
        prog="distill-run",
        description="Black-box and white-box distillation behind one entrypoint.",
    )
    for param in P.PARAMS:
        kwargs: dict[str, Any] = {"dest": param.dest, "help": param.help, "default": None}
        if param.repeatable:
            kwargs["action"] = "append"
        elif param.kind == "bool":
            kwargs["action"] = argparse.BooleanOptionalAction
        elif param.kind == "int":
            kwargs["type"] = int
        elif param.kind == "float":
            kwargs["type"] = float
        else:
            kwargs["type"] = str
        if param.engines != P.ALL:
            kwargs["help"] += f" [{', '.join(param.engines)}]"
        parser.add_argument(*param.flags, **kwargs)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv[:1] == ["doctor"]:
        from .doctor import main as doctor_main

        return doctor_main(raw_argv[1:])
    args = build_parser().parse_args(raw_argv)
    try:
        list_models = bool(args.list_models) or _env_enabled("DISTILL_LIST_MODELS")
        list_datasets = bool(args.list_datasets) or _env_enabled("DISTILL_LIST_DATASETS")
        list_engines = bool(args.list_engines) or _env_enabled("DISTILL_LIST_ENGINES")
        if list_models or list_datasets or list_engines:
            from .catalog import print_catalog

            print_catalog(
                models=list_models,
                datasets=list_datasets,
                engines=list_engines,
            )
            return int(ExitCode.OK)
        if args.show_default_config:
            engine_name = args.engine or os.environ.get("DISTILL_ENGINE")
            if engine_name not in P.ENGINES:
                raise UsageError(
                    f"--show-default-config requires --engine ({', '.join(P.ENGINES)})"
                )
            print(yaml.safe_dump(engine_defaults(engine_name), allow_unicode=True, sort_keys=False))
            return int(ExitCode.OK)
        engine_name = args.engine or os.environ.get("DISTILL_ENGINE")
        if engine_name:
            relaunched = maybe_run_in_engine_environment(engine_name, raw_argv)
            if relaunched is not None:
                return relaunched
        ctx = build_context(vars(args))
        launched = launcher.launch_trl_if_needed(ctx, raw_argv)
        if launched is not None:
            return launched
        launcher.configure_single_device(ctx)
        result = execute(ctx)
        logger.info("run %s succeeded: %s", ctx.run_id, result.message or "done")
        return int(ExitCode.OK)
    except DistillRunError as exc:
        logger.error("[%s] %s", exc.code, exc.message)
        return int(exc.exit_code)
    except KeyboardInterrupt:
        cancelled = CancelledError()
        logger.error("[%s] %s", cancelled.code, cancelled.message)
        return int(cancelled.exit_code)
    except Exception as exc:  # noqa: BLE001 - last resort, still exit cleanly
        logger.exception("unexpected failure")
        wrapped = classify(exc)
        return int(wrapped.exit_code)


def build_context(args: dict[str, Any]) -> RunContext:
    env = dict(os.environ)

    engine_name = args.get("engine") or env.get("DISTILL_ENGINE")
    if not engine_name:
        raise UsageError("--engine is required (or set DISTILL_ENGINE)")
    if engine_name not in P.ENGINES:
        raise UsageError(
            f"unknown engine '{engine_name}'; supported engines: {', '.join(P.ENGINES)} "
            "(run `distill-run --engines` for details)"
        )
    args["engine"] = engine_name

    params = merge_sources(engine_name, args, env)
    deferred = check_required(params)

    run_id = params.get("run_id") or generate_run_id()
    work_dir = Path(params.get("work_dir", P.DEFAULT_WORK_DIR))
    cache_dir = Path(params.get("dataset_cache_dir") or work_dir / "cache")
    output = Path(params.get("output"))
    prepare_environment(work_dir, cache_dir)

    config_env = expansion_env(params, env)
    config = engine_defaults(engine_name)
    config = deep_merge(config, load_config(params.get("config"), config_env))
    config = deep_merge(config, load_inline_config(params.get("config_json"), config_env))
    config = apply_config_overrides(config, params.get("config_overrides"))
    config = apply_param_config_overrides(config, params)

    logger.info("engine=%s run_id=%s output=%s", engine_name, run_id, output)
    if deferred:
        logger.info("expecting from --config: %s", ", ".join(p.flag for p in deferred))

    return RunContext(
        engine=engine_name,
        run_id=run_id,
        params=params,
        config=config,
        work_dir=work_dir,
        output=output,
        cache_dir=cache_dir,
        cancellation=install_handlers(),
    )


def execute(ctx: RunContext) -> RunResult:
    engine: Engine = registry.load(ctx.engine)

    preflight.check_writable(ctx.output_dir, "output")
    preflight.check_writable(ctx.work_dir, "work dir")
    preflight.check_space(ctx.output_dir)

    specs = engine.dataset_specs(ctx)
    engine.preflight(ctx)

    for dest, spec in specs.items():
        ctx.datasets[dest] = resolve(spec, ctx.cache_dir)

    if ctx.cancellation.requested:
        raise CancelledError("cancelled before the engine started")

    result = engine.run(ctx)
    for artifact in result.artifacts:
        logger.info("artifact: %s", artifact)
    return result


if __name__ == "__main__":
    sys.exit(main())
