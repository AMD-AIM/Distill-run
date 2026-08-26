# distill-run

One command for knowledge distillation, in two flavours:

- **black box** — a teacher served over an OpenAI-compatible API generates SFT
  data, then the student trains on it (EasyDistill + ms-swift)
- **white box** — teacher and student weights are loaded directly and the student
  learns from the teacher's logits (TRL `DistillationTrainer`), with FSDP2 or
  DeepSpeed ZeRO-3 for multi-GPU runs

The repository is self-contained: the custom TRL and EasyDistill revisions are
pinned Git submodules under `third_party/`. Heavy engines run in isolated virtual
environments, while the core CLI and test suite remain usable without Torch.

## Quick start

```bash
git clone --recursive <distill-run-repository>
cd distill-run
make dev PROFILE=rocm714

./run.sh install                   # start distill2 and install the frameworks
./run.sh teacher-check             # is the teacher API answering?
./run.sh easydistill-swift         # generate SFT data, then train the student
```

Paths, GPUs, model locations and the teacher URL all live in `env.sh`; nothing is
hard-coded in the Python.

## Commands

| Command | What it does |
| --- | --- |
| `./run.sh up` / `down` / `shell` / `status` | `distill2` container lifecycle |
| `./run.sh install` | installs EasyDistill, TRL, ms-swift and distill-run in the container |
| `./run.sh generate` | seed instructions → teacher API → SFT JSONL |
| `DATASET=<jsonl> ./run.sh sft` | trains the student on an existing SFT JSONL |
| `./run.sh easydistill-swift` | EasyDistill generation + Swift training |
| `./run.sh trl` | white-box distillation from both weight sets |
| `./run.sh trl-fsdp` | FSDP2-sharded white-box distillation across training GPUs |
| `./run.sh trl-zero3` | ZeRO-3 white-box distillation for teachers larger than one GPU |
| `./run.sh test` | unit tests plus the CLI smoke script, on the host |
| `./run.sh smoke` | tiny end-to-end run in the container (needs GPUs) |
| `./run.sh cli ...` | raw `distill-run` invocation inside the container |

Discover the currently supported capabilities without supplying `--engine`:

```bash
distill-run --models
distill-run --datasets
distill-run --engines
```

These commands print YAML suitable for both humans and scripts. Unknown engine
names, missing model paths, and unsupported dataset inputs include the
corresponding supported names or discovered local paths in their error message.

## Calling the CLI directly

```bash
distill-run --engine easydistill-swift \
  --teacher-base-url http://127.0.0.1:8000/v1 \
  --teacher-model-id /models/Qwen3.8-27B \
  --input examples/seed_instructions.jsonl \
  --model /models/Qwen3-0.6B \
  --output /output/run-1 \
  --temperature 0.7 --max-tokens 1024 --max-workers 4 \
  --lora-rank 8 --lora-alpha 32 \
  --num-train-epochs 3 --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 --learning-rate 1e-4 \
  --max-length 2048
```

Values resolve in the order **command-line argument > environment variable >
config file**, so `env.sh` can set the constants and a command line only names
what differs. Each engine accepts only the parameters that make sense for it:
passing `--teacher-base-url` to `--engine trl` is a usage error rather than a
silently ignored flag.

External files are optional. Each engine has a built-in production default
config, and any field can be supplied in the command:

```bash
distill-run --engine trl --show-default-config

distill-run --engine trl \
  --teacher /models/teacher --student /models/student \
  --dataset /data/prompts.jsonl --output /output/trl \
  --max-steps 500 \
  --learning-rate 2e-5 \
  --lora-rank 16
```

For an API that prefers one argument, pass the complete object with
`--config-json '{"trl":{"max_steps":500},"lora":{"r":16}}'`. `--set` remains
available for rare framework-specific keys. Precedence is: built-in defaults,
then `--config`, then `--config-json`, then `--set`, then explicit flags such as
`--learning-rate`.

Run `distill-run --help` for the authoritative list. See
[`docs/cli-parameters.md`](docs/cli-parameters.md) for the parameter ownership,
defaults and compatibility tables, [`docs/datasets.md`](docs/datasets.md) for
Hub/Parquet normalization, and `docs/engines.md` for engine behavior.

> **Temporary limitation:** single-machine multi-GPU entrypoints are disabled
> while isolated engine environments are being adapted. The underlying launcher
> code and configs are retained; use `--num-processes 1`.

Single-machine distribution support is retained in the codebase:

Omitting `--num-processes` defaults to one process and exposes only logical GPU
0 to the training framework. Use `--device N` to select another logical GPU.

```bash
# ms-swift uses torchrun internally
distill-run --engine swift --num-processes 4 \
  --student /models/student --dataset /output/sft.jsonl \
  --output /output/swift-4gpu --learning-rate 1e-4 --lora-rank 8

# TRL relaunches distill-run ranks under Accelerate
distill-run --engine trl --num-processes 6 --distributed-strategy zero3 \
  --teacher /models/teacher --student /models/student \
  --dataset examples/prompts.jsonl --output /output/trl-6gpu \
  --max-steps 100 --learning-rate 1e-5 --lora-rank 8
```

Built-in strategies are `fsdp2`, `zero1`, `zero2`, and `zero3`. Use
`--accelerate-config path/to/accelerate.yaml` for a custom launcher config.

## Exit codes

Failures are separated so a shell script can tell a typo from a dead teacher.

| Code | Meaning |
| --- | --- |
| 0 | success |
| 2 | bad or missing arguments |
| 3 | config file invalid |
| 4 | dataset content invalid |
| 5 | dataset could not be fetched |
| 6 | teacher endpoint unreachable |
| 7 | preflight failed (weights missing, disk full, …) |
| 8 | engine failed |
| 9 | out of memory |
| 130 | cancelled by Ctrl-C / SIGTERM |

## How a run is put together

```text
parse args ─► merge ─► load config ─► resolve data/models ─► preflight ─► engine
                                       container-local cache                    │
                                                                    the expensive part
```

Everything before the engine is cheap, which is the whole point: a wrong path, an
unreachable teacher or a mistyped hyper-parameter name fails in seconds instead of
after minutes of weight loading. Preflight checks hyper-parameter names against
the *installed* framework's argument class, because those names drift between
versions — ms-swift renamed `train_type` to `tuner_type`, and the framework itself
only notices after the model is in memory.

Dataset arguments are *specifications*, not necessarily existing files: a local
path, `https://…`, `hf://org/name` or `ms://org/name` all work, and network
sources are cached under `<work-dir>/cache` keyed by URI and revision. Engines
only ever see local paths.

Model arguments work the same way for training engines: `--student` and TRL's
`--teacher` accept a local path, `hf://org/model`, or `ms://org/model`. Hub
weights are downloaded at runtime into `<work-dir>/models`; use
`--model-cache-dir` and `--model-revision` to override that location or pin a
revision. This allows a Kubernetes Pod to run without shared storage, but its
datasets, weights, and checkpoints disappear when the Pod is deleted unless they
are copied or uploaded first. The `/models`, `/data`, and `/output` path
conventions remain available for deployments that can mount volumes.

```text
src/distill_run/
  cli.py          entrypoint: parse, dispatch, single exit funnel
  params.py       the parameter table — the parser, validation and per-engine
                  accepted sets are all derived from it
  config.py       YAML/JSON loading, ${ENV} expansion, value precedence
  dataset.py      URI → local path, with a content-addressed cache
  models.py       local/Hub model URI → container-local checkpoint
  fetchers.py     one class per scheme: local, http, hf, modelscope
  preflight.py    cheap checks: writable dirs, disk, weights, teacher, arg names
  context.py      what an engine is allowed to know about the run
  errors.py       exit codes and the single exception type
  launcher.py     multi-GPU: only distributed when explicitly asked
  signals.py      Ctrl-C → stop at a step boundary, then save
  utils.py        atomic writes, JSONL, HTTP session, child processes
  engines/
    base.py       the interface: dataset_specs → preflight → run
    easydistill.py  black box, stage one: teacher API → SFT JSONL
    swift.py        black box, stage two: student SFT
    easydistill_swift.py  both stages chained, with the intermediate JSONL kept
    trl.py          white box, single-GPU or FSDP2-sharded
    noop.py         no frameworks, used to test the CLI itself
```

Adding an engine means adding one module plus a line in
`engines/registry.py`; adding a parameter means editing `params.py` and nothing
else.

## Development

```bash
make dev PROFILE=rocm714  # submodules + isolated core/engine venvs
make doctor         # verify every engine environment
make check          # lint + unit tests + doctor
make test           # unit tests, no GPU needed
make lint           # ruff
./scripts/smoke-cli.sh   # exit-code contract, no GPU, no network
```

`make dev` uses only standard `venv` and pip operations. It initializes the
submodules, preserves the image's preinstalled ROCm Torch, installs the core in
`.venv`, and creates `.venvs/swift`, `.venvs/trl`, and
`.venvs/easydistill`. Use `ENGINES=trl` to install only one engine. The selected
profile and exact submodule commits are recorded in
`.distill/environment.lock.json`.

`pip install -e .` remains valid for installing only the lightweight core. It
does not perform network or Git operations.
